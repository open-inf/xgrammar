"""Minimum-only GLM XML strings keep Unicode counts and bounded parser work."""
import json

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def tag(minimum, *, maximum=None, style='glm_xml'):
    value = dict(type='string', minLength=minimum)
    if maximum is not None:
        value['maxLength'] = maximum
    schema = dict(type='object', properties=dict(value=value, following=dict(type='string')),
                  required=['value', 'following'], additionalProperties=False)
    return json.dumps(dict(type='structural_tag', format=dict(type='tag',
        begin='<tool_call>record', content=dict(type='json_schema', style=style,
                                              json_schema=schema), end='</tool_call>')))


def wire(value):
    return ('<tool_call>record<arg_key>value</arg_key><arg_value>' + value +
            '</arg_value><arg_key>following</arg_key><arg_value>next</arg_value></tool_call>')


@pytest.mark.parametrize('minimum', [1, 40, 160])
def test_minimum_only_raw_string_counts_unicode_codepoints(minimum):
    grammar = xgr.Grammar.from_structural_tag(tag(minimum))
    for value, valid in [('', False), ('x' * (minimum - 1), False),
                         ('x' * minimum, True), ('é' * minimum, True),
                         ('😀' * minimum, True), ('\n' * minimum, True),
                         ('é😀\n' * minimum, True)]:
        assert _is_grammar_accept_string(grammar, wire(value)) is valid


@pytest.mark.parametrize('minimum', [40, 160])
def test_minimum_only_raw_string_masks_partial_unicode_and_rollback(minimum):
    alphabet = ['<eos>', *map(chr, range(32, 127)), b'\xc3', b'\xa9', '😀',
                'xx', '</arg_value>', '</tool_call>']
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(
        alphabet, xgr.VocabType.RAW, stop_token_ids=[0]))
    matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(tag(minimum)))
    assert matcher.accept_string('<tool_call>record<arg_key>value</arg_key><arg_value>' +
                                 'x' * (minimum - 1))
    mask = xgr.allocate_token_bitmask(1, len(alphabet))
    closer = alphabet.index('</arg_value>')
    for token in (alphabet.index(b'\xc3'), alphabet.index(b'\xa9')):
        matcher.fill_next_token_bitmask(mask, 0)
        assert int(mask[0, token // 32]) & (1 << (token % 32))
        state = matcher._debug_print_internal_state()
        assert matcher.accept_token(token)
        matcher.rollback(1)
        assert matcher._debug_print_internal_state() == state
        assert matcher.accept_token(token)
    matcher.fill_next_token_bitmask(mask, 0)
    assert int(mask[0, closer // 32]) & (1 << (closer % 32))


def test_long_raw_value_does_not_accumulate_minimum_prefix_alternatives():
    alphabet = ['<eos>', *map(chr, range(32, 127)), '</arg_value>', '</tool_call>']
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(
        alphabet, xgr.VocabType.RAW, stop_token_ids=[0]))
    matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(tag(160)))
    assert matcher.accept_string('<tool_call>record<arg_key>value</arg_key><arg_value>' +
                                 'x' * 4096)
    assert matcher._debug_print_internal_state().count('ParserState(') < 16


def test_finite_bounds_preserve_their_language():
    grammar = xgr.Grammar.from_structural_tag(tag(3, maximum=8))
    for value, valid in [('xx', False), ('é😀x', True), ('x' * 8, True), ('x' * 9, False)]:
        assert _is_grammar_accept_string(grammar, wire(value)) is valid
