"""Portable diagnostic receipts: preserve numbers and hashes while removing paths."""
import hashlib
import ipaddress
import json
import re


PRIVATE_HOME = re.compile(r'/(?:data/)?home/[^/\s]+/|/Users/[^/\s]+/')
IPV4 = re.compile(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])')


def byte_digest(data):
    """Hash the exact captured bytes, before any dtype conversion."""
    return hashlib.sha256(data).hexdigest()


def require_shareable(value):
    """Reject residual machine locations; never erase numerical JSON fields."""
    text = json.dumps(value, allow_nan=False)
    if PRIVATE_HOME.search(text):
        raise ValueError('Private home path remains in evidence')
    for match in IPV4.finditer(text):
        try:
            ipaddress.ip_address(match.group())
        except ValueError:
            continue
        raise ValueError('Machine address remains in evidence')
    return value


def redact_locations(value, replacements):
    """Apply explicit location mappings only to text; fail closed on leftovers.

    Replacements come from ignored machine configuration, never this source.
    Longest prefixes win. Numeric values, booleans, None and hashes survive.
    """
    mappings = sorted(replacements.items(), key=lambda item:len(item[0]), reverse=True)
    if any(not old or not isinstance(old,str) or not isinstance(new,str) for old,new in mappings):
        raise ValueError('Location mappings must be nonempty strings')
    def transform(item):
        if isinstance(item,str):
            for old,new in mappings:
                item=item.replace(old,new)
            return item
        if isinstance(item,list):
            return [transform(x) for x in item]
        if isinstance(item,dict):
            pairs=[(transform(k),transform(v)) for k,v in item.items()]
            result=dict(pairs)
            if len(result)!=len(item):
                raise ValueError('Redaction would merge distinct record keys')
            return result
        return item
    return require_shareable(transform(value))
