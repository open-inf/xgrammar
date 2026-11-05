"""GLM raw word-count patterns preserve whitespace and Unicode."""
import json

import pytest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def tag(pattern, style='glm_xml'):
    schema = dict(type='object', properties=dict(summary=dict(type='string', pattern=pattern),
                  command=dict(type='string')), required=['summary', 'command'],
                  additionalProperties=False)
    return json.dumps(dict(type='structural_tag', format=dict(type='tag',
        begin='<tool_call>execute', content=dict(type='json_schema', style=style,
        json_schema=schema), end='</tool_call>')))


@pytest.mark.parametrize('value,valid', [
    ('Some words here', True), ('Some\u00a0words\u00a0here', True),
    ('Some 中文 words', True), ('Some words', False),
    ('lower case words', False), ('One two three four five six seven eight nine', False),
])
def test_raw_whitespace_pattern_preserves_language(value, valid):
    grammar = xgr.Grammar.from_structural_tag(tag(r'^[A-Z]\S*(?:\s+\S+){2,7}$'))
    assert 'Regex(' in str(grammar)
    wire = ('<tool_call>execute<arg_key>summary</arg_key><arg_value>' + value +
            '</arg_value><arg_key>command</arg_key><arg_value>echo ok</arg_value></tool_call>')
    assert _is_grammar_accept_string(grammar, wire) is valid


def test_fixed_prefix_and_other_xml_patterns_keep_the_existing_path():
    assert 'Regex(' not in str(xgr.Grammar.from_structural_tag(tag(r'^wf_[a-z0-9-]{6,}$')))
    assert 'Regex(' not in str(xgr.Grammar.from_structural_tag(
        tag(r'^[A-Z]\S*(?:\s+\S+){2,7}$', style='qwen_xml')))
    for pattern in [r'[\sA-Z]+', r'[\S]+', r'\\s+']:
        assert 'Regex(' not in str(xgr.Grammar.from_structural_tag(tag(pattern)))


def test_raw_pattern_masks_allow_partial_unicode_and_rollback():
    spec = tag(r'^[A-Z]\S*(?:\s+\S+){2,7}$')
    vocab = [b'<eos>', b'A', b' ', b'\xc2', b'\xa0', b'\xe4', b'\xb8', b'\xad', b'x']
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, stop_token_ids=[0]))
    matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(spec), max_rollback_tokens=1)
    assert matcher.accept_string('<tool_call>execute<arg_key>summary</arg_key><arg_value>A')
    mask = xgr.allocate_token_bitmask(1, len(vocab))
    for token in [3, 4, 5, 6, 7, 2, 8]:
        matcher.fill_next_token_bitmask(mask, 0)
        assert int(mask[0, token // 32]) & (1 << (token % 32))
        before = matcher._debug_print_internal_state()
        assert matcher.accept_token(token)
        matcher.rollback(1)
        assert matcher._debug_print_internal_state() == before
        assert matcher.accept_token(token)
