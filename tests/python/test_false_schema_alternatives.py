"""Never schemas must not become unconstrained alternatives in decoder grammars."""
import json

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


OBJECT = {
    'type': 'object',
    'properties': {'value': {'type': 'string'}},
    'required': ['value'],
    'additionalProperties': False,
}


@pytest.mark.parametrize('union', ['anyOf', 'oneOf'])
@pytest.mark.parametrize('never', [False, {'not': {}}, {'not': True}])
def test_never_alternatives_preserve_only_the_valid_languages(union, never):
    schema = {
        'type': 'object',
        'properties': {'rows': {'type': 'array', 'items': {union: [never, OBJECT, {'type': 'null'}]}}},
        'required': ['rows'],
        'additionalProperties': False,
    }
    grammar = xgr.Grammar.from_json_schema(schema)
    for value, expected in [
        ({'rows': []}, True),
        ({'rows': [{'value': 'ok'}, None]}, True),
        ({'rows': [['a', 'b']]}, False),
        ({'rows': [123]}, False),
        ({'rows': [{}]}, False),
        ({'rows': [{'value': 123}]}, False),
        ({'rows': [{'value': 'ok', 'extra': True}]}, False),
    ]:
        assert _is_grammar_accept_string(grammar, json.dumps(value)) is expected


@pytest.mark.parametrize('schema', [
    {'not': {}}, {'not': True},
    {'anyOf': [False, {'not': {}}]},
    {'oneOf': [False, {'not': True}]},
    {'allOf': [{'type': 'string'}, {'not': {}}]},
])
def test_fully_impossible_schemas_are_rejected(schema):
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_json_schema(schema)


def test_nested_impossible_union_does_not_poison_a_valid_alternative():
    grammar = xgr.Grammar.from_json_schema({
        'anyOf': [{'anyOf': [False, {'not': {}}]}, {'type': 'integer'}],
    })
    assert _is_grammar_accept_string(grammar, '12')
    assert not _is_grammar_accept_string(grammar, '[]')


def test_invalid_alternatives_are_not_silently_dropped():
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_json_schema({'anyOf': [{'type': 123}, {'type': 'null'}]})


def test_schema_looking_literal_values_are_unchanged():
    literal = {'anyOf': [{'not': {}}]}
    grammar = xgr.Grammar.from_json_schema({'const': literal})
    assert _is_grammar_accept_string(grammar, json.dumps(literal, separators=(",", ":")))
    assert not _is_grammar_accept_string(grammar, '{}')


def test_negating_false_keeps_the_universal_language():
    grammar = xgr.Grammar.from_json_schema({'not': False})
    for value in [None, True, 123, 'text', [], {}]:
        assert _is_grammar_accept_string(grammar, json.dumps(value))


def test_draft7_ref_sibling_handling_is_preserved():
    grammar = xgr.Grammar.from_json_schema({
        '$schema': 'http://json-schema.org/draft-07/schema#',
        '$ref': '#/definitions/value',
        'not': {},
        'definitions': {'value': {'type': 'string'}},
    })
    assert _is_grammar_accept_string(grammar, '"ok"')
    assert not _is_grammar_accept_string(grammar, '12')
