/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/tvm_ffi/tvm_ffi.cc
 * \brief TVM-FFI bindings for xgrammar.
 */

#include <dlpack/dlpack.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/tvm_ffi.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include "../support/thread_pool.h"
#include <string>
#include <variant>
#include <vector>

#include "../grammar_functor.h"
#include "../json_schema_converter.h"
#include "../regex_converter.h"
#include "../support/utils.h"
#include "../testing.h"
#include "python_methods.h"
#include "xgrammar/exception.h"
#include "xgrammar/matcher.h"

namespace ffi = tvm::ffi;
namespace refl = tvm::ffi::reflection;

namespace xgrammar {

// ----- Error handling -----

// ----- Helpers: convert FFI types to/from xgrammar types -----

static std::string BytesToString(const tvm::ffi::Bytes& bytes) {
  std::string result(bytes.data(), bytes.size());
  return result;
}

// Convert ffi::Array<ffi::Any> to vector<string>; each element can be str or bytes (like
// accept_string).
static std::vector<std::string> ArrayAnyToVectorString(ffi::Array<ffi::Any> array) {
  std::vector<std::string> result;
  result.reserve(static_cast<size_t>(array.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(array.size()); ++i) {
    ffi::AnyView view = array[i];
    if (view.as<ffi::Bytes>()) {
      result.push_back(BytesToString(view.cast<ffi::Bytes>()));
    } else if (view.as<ffi::String>()) {
      result.push_back(view.cast<ffi::String>());
    } else {
      TVM_FFI_THROW(RuntimeError) << "Unsupported type in encoded_vocab: expected str or bytes";
      XGRAMMAR_UNREACHABLE();
    }
  }
  return result;
}

static ffi::Array<ffi::Bytes> VectorStringToBytesArray(const std::vector<std::string>& string_vector
) {
  ffi::Array<ffi::Bytes> bytes_array;
  for (const auto& value : string_vector) {
    bytes_array.push_back(ffi::Bytes(value));
  }
  return bytes_array;
}

static std::optional<int64_t> OptionalIntFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  return v.cast<int64_t>();
}

static std::optional<bool> OptionalBoolFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  return static_cast<bool>(v.cast<int64_t>());
}

static std::optional<std::vector<int32_t>> OptionalInt32VectorFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  ffi::Array<int64_t> array = v.cast<ffi::Array<int64_t>>();
  std::vector<int32_t> result;
  result.reserve(static_cast<size_t>(array.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(array.size()); ++i)
    result.push_back(static_cast<int32_t>(array[i]));
  return result;
}

static std::optional<std::vector<int>> OptionalIntVectorFromView(ffi::AnyView v) {
  if (v == nullptr) return std::nullopt;
  ffi::Array<int64_t> array = v.cast<ffi::Array<int64_t>>();
  std::vector<int> result;
  result.reserve(static_cast<size_t>(array.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(array.size()); ++i)
    result.push_back(static_cast<int>(array[i]));
  return result;
}

static std::optional<std::pair<std::string, std::string>> OptionalSeparatorsFromView(ffi::AnyView v
) {
  if (v == nullptr) return std::nullopt;
  ffi::Array<ffi::String> separators_array = v.cast<ffi::Array<ffi::String>>();
  if (separators_array.size() < 2) return std::nullopt;
  return std::make_pair(separators_array[0], separators_array[1]);
}

static std::variant<std::string, int32_t> ParseMaxThreads(ffi::AnyView max_threads_view) {
  if (max_threads_view == nullptr) return "auto";
  auto int_value = max_threads_view.as<int64_t>();
  if (int_value.has_value()) {
    return static_cast<int32_t>(int_value.value());
  }
  auto string_value = max_threads_view.as<ffi::String>();
  if (string_value.has_value()) {
    return string_value.value();
  }
  TVM_FFI_THROW(RuntimeError) << "Invalid max_threads value";
  XGRAMMAR_UNREACHABLE();
}

// Wrap std::exception into TVM-FFI error
#define XGRAMMAR_FFI_TRY_BEGIN() try {
#define XGRAMMAR_FFI_TRY_END()                   \
  }                                              \
  catch (const XGrammarError& e) {               \
    throw ffi::Error(e.GetType(), e.what(), ""); \
    XGRAMMAR_UNREACHABLE();                      \
  }

// ----- Object wrappers (hold xgrammar types, inherit ffi::Object) -----

class TokenizerInfoObj : public ffi::Object {
 public:
  TokenizerInfo value;

  explicit TokenizerInfoObj(TokenizerInfo v) : value(std::move(v)) {}

  TokenizerInfoObj(
      ffi::Array<ffi::Any> encoded_vocab,
      int64_t vocab_type,
      ffi::AnyView vocab_size_opt,
      ffi::AnyView stop_token_ids_opt,
      bool add_prefix_space
  )
      : value(NullObj{}) {
    XGRAMMAR_FFI_TRY_BEGIN();
    value = TokenizerInfo_Init(
        ArrayAnyToVectorString(encoded_vocab),
        static_cast<int>(vocab_type),
        OptionalIntFromView(vocab_size_opt),
        OptionalInt32VectorFromView(stop_token_ids_opt),
        add_prefix_space
    );
    XGRAMMAR_FFI_TRY_END();
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.TokenizerInfo", TokenizerInfoObj, ffi::Object
  );
};

class GrammarObj : public ffi::Object {
 public:
  Grammar value;

  explicit GrammarObj(Grammar v) : value(std::move(v)) {}

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("xgrammar.tvm_ffi_binding.Grammar", GrammarObj, ffi::Object);
};

static std::vector<NamedGrammar> NamedGrammarsFromViews(
    ffi::AnyView names_view, ffi::AnyView values_view
) {
  auto names = names_view.cast<ffi::Array<ffi::Any>>();
  auto values = values_view.cast<ffi::Array<ffi::Any>>();
  if (names.size() != values.size()) {
    throw XGrammarError("Named grammar names and values must have the same length");
  }
  std::vector<NamedGrammar> named_grammars;
  named_grammars.reserve(static_cast<size_t>(names.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(names.size()); ++i) {
    std::string name = names[i].cast<ffi::String>();
    ffi::AnyView value = values[i];
    if (const auto* grammar_obj = value.as<GrammarObj>()) {
      named_grammars.push_back({std::move(name), grammar_obj->value});
    } else if (value.as<ffi::String>()) {
      named_grammars.push_back({std::move(name), std::string(value.cast<ffi::String>())});
    } else {
      throw XGrammarError("Named grammar values must be Grammar or Lark strings");
    }
  }
  return named_grammars;
}

class CompiledGrammarObj : public ffi::Object {
 public:
  CompiledGrammar value;

  explicit CompiledGrammarObj(CompiledGrammar v) : value(std::move(v)) {}

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.CompiledGrammar", CompiledGrammarObj, ffi::Object
  );
};

class GrammarCompilerObj : public ffi::Object {
 public:
  GrammarCompiler value;

  explicit GrammarCompilerObj(GrammarCompiler v) : value(std::move(v)) {}

  GrammarCompilerObj(
      ffi::ObjectRef tokenizer_ref,
      int64_t max_threads,
      bool cache_enabled,
      int64_t max_memory_bytes
  )
      : value(
            tokenizer_ref.as<TokenizerInfoObj>()->value,
            static_cast<int>(max_threads),
            cache_enabled,
            max_memory_bytes
        ) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.GrammarCompiler", GrammarCompilerObj, ffi::Object
  );
};

class GrammarMatcherObj : public ffi::Object {
 public:
  GrammarMatcher value;

  explicit GrammarMatcherObj(GrammarMatcher v) : value(std::move(v)) {}

  GrammarMatcherObj(
      ffi::ObjectRef compiled_grammar_ref,
      ffi::AnyView override_stop_tokens_opt,
      bool terminate_without_stop_token,
      int64_t max_rollback_tokens,
      std::optional<float> default_temperature
  )
      : value(
            compiled_grammar_ref.as<CompiledGrammarObj>()->value,
            OptionalIntVectorFromView(override_stop_tokens_opt),
            terminate_without_stop_token,
            static_cast<int>(max_rollback_tokens),
            default_temperature
        ) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.GrammarMatcher", GrammarMatcherObj, ffi::Object
  );
};

class BatchGrammarMatcherObj : public ffi::Object {
 public:
  BatchGrammarMatcher value;

  BatchGrammarMatcherObj() = default;
  explicit BatchGrammarMatcherObj(BatchGrammarMatcher v) : value(std::move(v)) {}

  explicit BatchGrammarMatcherObj(ffi::AnyView max_threads_view)
      : value(ParseMaxThreads(max_threads_view)) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "xgrammar.tvm_ffi_binding.BatchGrammarMatcher", BatchGrammarMatcherObj, ffi::Object
  );
};

// ----- Registration: ObjectDef -----
// Custom constructors are handled via lambda wrappers below; TVM-FFI uses refl::init<Args...>()
// and we need to bridge FFI types to xgrammar types in those lambdas.

// Both entry points use the same scalar parser operations and rollback path.
template <typename Token, typename Words, typename Keeps>
static void TraverseFilteredChain(
    GrammarMatcher& matcher, const Token* tokens, DLTensor* masks, float* temps,
    const int32_t* supplied, const Words& words, const Keeps& keeps, int64_t minimum,
    const Words& min_words, const Keeps& min_keeps) {
  const int64_t queries = masks->shape[0], width = masks->shape[1];
  auto* bits = static_cast<int32_t*>(masks->data);
            auto apply = [](int32_t* row, auto w, auto k) {
              for (int64_t i = 0; i < static_cast<int64_t>(w.size()); ++i) {
                row[w[i]] &= static_cast<int32_t>(k[i]);
              }
            };
            auto clear = [bits, width, queries](int64_t start) {
              std::fill(bits + start * width, bits + queries * width, 0);
            };
            int advanced = 0;
            try {
              for (int64_t query = 0; query < queries; ++query) {
                if (matcher.IsTerminated()) { clear(query); break; }
                auto* row = bits + query * width;
                if (query == 0 && supplied) {
                  std::copy_n(supplied, width, row);
                } else {
                  matcher.FillNextTokenBitmask(masks, query);
                  if (!matcher.IsCompleted()) { apply(row, words, keeps); }
                }
                if (auto temperature = matcher.GetTemperature()) { temps[query] = *temperature; }
                if (query < minimum) { apply(row, min_words, min_keeps); }
                if (query + 1 < queries) {
                  const int32_t token = tokens[query + 1];
                  if (!(static_cast<uint32_t>(row[token / 32]) & (uint32_t{1} << (token % 32)))) {
                    clear(query + 1); break;
                  }
                  if (!matcher.AcceptToken(token)) {
                    TVM_FFI_THROW(RuntimeError) << "Speculative grammar mask disagrees with matcher";
                  }
                  ++advanced;
                }
              }
            } catch (...) {
              if (advanced) matcher.Rollback(advanced);
              throw;
            }
            if (advanced) matcher.Rollback(advanced);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using O = ffi::ObjectRef;

  // TokenizerInfo: init(encoded_vocab, vocab_type, vocab_size_opt, stop_token_ids_opt,
  // add_prefix_space)
  refl::ObjectDef<TokenizerInfoObj>()
      .def(refl::init<ffi::Array<ffi::Any>, int64_t, ffi::AnyView, ffi::AnyView, bool>())
      .def(
          "vocab_type",
          [](const TokenizerInfoObj* o) {
            return static_cast<int64_t>(TokenizerInfo_GetVocabType(o->value));
          }
      )
      .def(
          "vocab_size",
          [](const TokenizerInfoObj* o) { return static_cast<int64_t>(o->value.GetVocabSize()); }
      )
      .def(
          "add_prefix_space", [](const TokenizerInfoObj* o) { return o->value.GetAddPrefixSpace(); }
      )
      .def(
          "decoded_vocab",
          [](const TokenizerInfoObj* o) {
            const auto& decoded_vocab = o->value.GetDecodedVocab();
            return VectorStringToBytesArray(decoded_vocab);
          }
      )
      .def(
          "stop_token_ids",
          [](const TokenizerInfoObj* o) {
            const auto& stop_token_ids = o->value.GetStopTokenIds();
            ffi::Array<int64_t> stop_token_ids_array;
            for (int32_t token_id : stop_token_ids)
              stop_token_ids_array.push_back(static_cast<int64_t>(token_id));
            return stop_token_ids_array;
          }
      )
      .def(
          "special_token_ids",
          [](const TokenizerInfoObj* o) {
            const auto& special_token_ids = o->value.GetSpecialTokenIds();
            ffi::Array<int64_t> special_token_ids_array;
            for (int32_t token_id : special_token_ids)
              special_token_ids_array.push_back(static_cast<int64_t>(token_id));
            return special_token_ids_array;
          }
      )
      .def(
          "dump_metadata",
          [](const TokenizerInfoObj* o) { return ffi::String(o->value.DumpMetadata()); }
      )
      .def_static(
          "from_vocab_and_metadata",
          [](ffi::Array<ffi::Any> encoded_vocab, ffi::String metadata) {
            XGRAMMAR_FFI_TRY_BEGIN();
            auto v = TokenizerInfo::FromVocabAndMetadata(
                ArrayAnyToVectorString(encoded_vocab), metadata
            );
            return ffi::ObjectRef(ffi::make_object<TokenizerInfoObj>(std::move(v)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "_detect_metadata_from_hf",
          [](ffi::String backend_str) {
            return ffi::String(TokenizerInfo::DetectMetadataFromHF(backend_str));
          }
      )
      .def(
          "serialize_json",
          [](const TokenizerInfoObj* o) { return ffi::String(o->value.SerializeJSON()); }
      )
      .def_static("deserialize_json", [](ffi::String json_string) {
        XGRAMMAR_FFI_TRY_BEGIN();
        auto r = TokenizerInfo::DeserializeJSON(json_string);
        if (std::holds_alternative<SerializationError>(r)) {
          const auto& err = std::get<SerializationError>(r);
          throw ffi::Error(GetTypeFromVariantError(err), GetMessageFromVariantError(err), "");
        }
        return ffi::ObjectRef(ffi::make_object<TokenizerInfoObj>(std::get<TokenizerInfo>(r)));
        XGRAMMAR_FFI_TRY_END();
      });

  // Grammar
  refl::ObjectDef<GrammarObj>()
      .def("to_string", [](const GrammarObj* o) { return ffi::String(o->value.ToString()); })
      .def_static(
          "from_ebnf",
          [](ffi::String ebnf_str, ffi::String root_rule_name) {
            return ffi::ObjectRef(
                ffi::make_object<GrammarObj>(Grammar::FromEBNF(ebnf_str, root_rule_name))
            );
          }
      )
      .def_static(
          "from_json_schema",
          [](ffi::String schema,
             bool any_whitespace,
             ffi::AnyView indent,
             ffi::AnyView separators,
             bool strict_mode,
             ffi::AnyView max_whitespace_cnt,
             bool print_converted_ebnf,
             bool any_order) {
            XGRAMMAR_FFI_TRY_BEGIN();
            auto g = Grammar::FromJSONSchema(
                schema,
                any_whitespace,
                OptionalIntFromView(indent),
                OptionalSeparatorsFromView(separators),
                strict_mode,
                OptionalIntFromView(max_whitespace_cnt),
                print_converted_ebnf,
                any_order
            );
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(g)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "from_regex",
          [](ffi::String regex, bool print_converted_ebnf) {
            return ffi::ObjectRef(
                ffi::make_object<GrammarObj>(Grammar::FromRegex(regex, print_converted_ebnf))
            );
          }
      )
      .def_static(
          "from_lark",
          [](ffi::String lark_string,
             ffi::AnyView tokenizer_info_view,
             ffi::AnyView named_grammar_names_view,
             ffi::AnyView named_grammar_values_view) {
            XGRAMMAR_FFI_TRY_BEGIN();
            std::optional<TokenizerInfo> tokenizer_info = std::nullopt;
            if (tokenizer_info_view != nullptr) {
              tokenizer_info =
                  tokenizer_info_view.cast<ffi::ObjectRef>().as<TokenizerInfoObj>()->value;
            }
            auto named_grammars =
                NamedGrammarsFromViews(named_grammar_names_view, named_grammar_values_view);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                Grammar::FromLark(lark_string, tokenizer_info, named_grammars)
            ));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "from_structural_tag",
          [](ffi::String structural_tag_json) {
            XGRAMMAR_FFI_TRY_BEGIN();
            Grammar grammar = Grammar_FromStructuralTag(structural_tag_json);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def_static(
          "builtin_json_grammar",
          []() {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(Grammar::BuiltinJSONGrammar()));
          }
      )
      .def_static(
          "union",
          [](ffi::Array<O> grammars) {
            std::vector<Grammar> grammar_list;
            grammar_list.reserve(static_cast<size_t>(grammars.size()));
            for (int64_t i = 0; i < static_cast<int64_t>(grammars.size()); ++i) {
              grammar_list.push_back(grammars[i].as<GrammarObj>()->value);
            }
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(Grammar::Union(grammar_list)));
          }
      )
      .def_static(
          "concat",
          [](ffi::Array<O> grammars) {
            std::vector<Grammar> grammar_list;
            grammar_list.reserve(static_cast<size_t>(grammars.size()));
            for (int64_t i = 0; i < static_cast<int64_t>(grammars.size()); ++i) {
              grammar_list.push_back(grammars[i].as<GrammarObj>()->value);
            }
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(Grammar::Concat(grammar_list)));
          }
      )
      .def(
          "serialize_json",
          [](const GrammarObj* o) { return ffi::String(o->value.SerializeJSON()); }
      )
      .def_static("deserialize_json", [](ffi::String json_string) {
        XGRAMMAR_FFI_TRY_BEGIN();
        Grammar grammar = Grammar_DeserializeJSON(json_string);
        return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
        XGRAMMAR_FFI_TRY_END();
      });

  // CompiledGrammar
  refl::ObjectDef<CompiledGrammarObj>()
      .def(
          "grammar",
          [](const CompiledGrammarObj* o) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(o->value.GetGrammar()));
          }
      )
      .def(
          "tokenizer_info",
          [](const CompiledGrammarObj* o) {
            return ffi::ObjectRef(ffi::make_object<TokenizerInfoObj>(o->value.GetTokenizerInfo()));
          }
      )
      .def(
          "memory_size_bytes",
          [](const CompiledGrammarObj* o) {
            return static_cast<int64_t>(o->value.MemorySizeBytes());
          }
      )
      .def(
          "serialize_json",
          [](const CompiledGrammarObj* o) { return ffi::String(o->value.SerializeJSON()); }
      )
      .def_static("deserialize_json", [](ffi::String json_string, O tokenizer_ref) {
        XGRAMMAR_FFI_TRY_BEGIN();
        const TokenizerInfo& tokenizer_info = tokenizer_ref.as<TokenizerInfoObj>()->value;
        CompiledGrammar compiled_grammar =
            CompiledGrammar_DeserializeJSON(json_string, tokenizer_info);
        return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(std::move(compiled_grammar)));
        XGRAMMAR_FFI_TRY_END();
      });

  // GrammarCompiler: init(tokenizer_info, max_threads, cache_enabled, max_memory_bytes)
  refl::ObjectDef<GrammarCompilerObj>()
      .def(refl::init<O, int64_t, bool, int64_t>())
      .def(
          "compile_json_schema",
          [](GrammarCompilerObj* o,
             ffi::String schema,
             bool any_whitespace,
             ffi::AnyView indent,
             ffi::AnyView separators,
             bool strict_mode,
             ffi::AnyView max_whitespace_cnt,
             bool any_order) {
            XGRAMMAR_FFI_TRY_BEGIN();
            CompiledGrammar cg = o->value.CompileJSONSchema(
                schema,
                any_whitespace,
                OptionalIntFromView(indent),
                OptionalSeparatorsFromView(separators),
                strict_mode,
                OptionalIntFromView(max_whitespace_cnt),
                any_order
            );
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(std::move(cg)));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def(
          "compile_builtin_json_grammar",
          [](GrammarCompilerObj* o) {
            return ffi::ObjectRef(
                ffi::make_object<CompiledGrammarObj>(o->value.CompileBuiltinJSONGrammar())
            );
          }
      )
      .def(
          "compile_structural_tag",
          [](GrammarCompilerObj* o, ffi::String structural_tag_json) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileStructuralTag(structural_tag_json)
            ));
          }
      )
      .def(
          "compile_regex",
          [](GrammarCompilerObj* o, ffi::String regex) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(o->value.CompileRegex(regex))
            );
          }
      )
      .def(
          "compile_lark",
          [](GrammarCompilerObj* o,
             ffi::String lark_string,
             ffi::AnyView named_grammar_names_view,
             ffi::AnyView named_grammar_values_view) {
            XGRAMMAR_FFI_TRY_BEGIN();
            auto named_grammars =
                NamedGrammarsFromViews(named_grammar_names_view, named_grammar_values_view);
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileLark(lark_string, named_grammars)
            ));
            XGRAMMAR_FFI_TRY_END();
          }
      )
      .def(
          "compile_grammar_ebnf",
          [](GrammarCompilerObj* o, O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileGrammar(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "compile_grammar_from_strings",
          [](GrammarCompilerObj* o, ffi::String ebnf_str, ffi::String root_rule_name) {
            return ffi::ObjectRef(ffi::make_object<CompiledGrammarObj>(
                o->value.CompileGrammar(ebnf_str, root_rule_name)
            ));
          }
      )
      .def("clear_cache", [](GrammarCompilerObj* o) { o->value.ClearCache(); })
      .def(
          "get_cache_size_bytes",
          [](const GrammarCompilerObj* o) { return o->value.GetCacheSizeBytes(); }
      )
      .def("cache_limit_bytes", [](const GrammarCompilerObj* o) {
        return o->value.CacheLimitBytes();
      });

  // BatchGrammarMatcher: init(max_threads)
  refl::ObjectDef<BatchGrammarMatcherObj>()
      .def(refl::init<ffi::AnyView>())
      .def(
          "batch_fill_next_token_bitmask",
          [](BatchGrammarMatcherObj* o,
             ffi::Array<O> matchers_ref,
             ffi::AnyView batch_token_bitmask,
             ffi::AnyView indices,
             bool debug_print) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            DLTensor* bitmask = batch_token_bitmask.cast<DLTensor*>();
            o->value.BatchFillNextTokenBitmask(
                &matchers, bitmask, OptionalInt32VectorFromView(indices), debug_print
            );
          }
      )
      .def_static(
          "batch_fill_temperature",
          [](ffi::Array<O> matchers_ref, ffi::AnyView temperatures, ffi::AnyView indices) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            DLTensor* temperatures_ptr = temperatures.cast<DLTensor*>();
            BatchGrammarMatcher::BatchFillTemperature(
                matchers, temperatures_ptr, OptionalInt32VectorFromView(indices)
            );
          }
      )
      .def_static(
          "batch_accept_string",
          [](ffi::Array<O> matchers_ref, ffi::Array<ffi::Any> input_str_byte_union, bool debug_print
          ) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            std::vector<std::string> input_strings = ArrayAnyToVectorString(input_str_byte_union);
            std::vector<uint8_t> acceptance_results =
                BatchGrammarMatcher::BatchAcceptString(&matchers, input_strings, debug_print);
            ffi::Array<int64_t> acceptance_results_array;
            for (uint8_t acceptance_flag : acceptance_results)
              acceptance_results_array.push_back(static_cast<int64_t>(acceptance_flag));
            return acceptance_results_array;
          }
      )
      .def_static(
          "batch_traverse_filtered_chains",
          [](ffi::Array<O> matchers_ref, ffi::Array<int64_t> rows,
             ffi::AnyView inputs_value, ffi::AnyView masks_value, ffi::AnyView temps_value,
             ffi::Array<ffi::Any> supplied_values,
             ffi::Array<ffi::Array<int64_t>> words, ffi::Array<ffi::Array<int64_t>> keeps,
             ffi::Array<int64_t> minimum, ffi::Array<ffi::Array<int64_t>> min_words,
             ffi::Array<ffi::Array<int64_t>> min_keeps, int64_t threads) {
            const int64_t count = rows.size();
            XGRAMMAR_CHECK(matchers_ref.size() == static_cast<size_t>(count) && supplied_values.size() == static_cast<size_t>(count)
                && words.size() == static_cast<size_t>(count) && keeps.size() == static_cast<size_t>(count) && minimum.size() == static_cast<size_t>(count)
                && min_words.size() == static_cast<size_t>(count) && min_keeps.size() == static_cast<size_t>(count));
            XGRAMMAR_CHECK(threads >= 1 && threads <= 8);
            if (!count) return;
            auto tensor = [](ffi::AnyView value, int ndim, int code, int bits) {
              auto* t = value.cast<DLTensor*>();
              XGRAMMAR_CHECK(t && t->device.device_type == kDLCPU && t->ndim == ndim
                  && t->dtype.code == code && t->dtype.lanes == 1 && t->byte_offset == 0);
              XGRAMMAR_CHECK(bits ? t->dtype.bits == bits :
                  (t->dtype.bits == 32 || t->dtype.bits == 64));
              int64_t stride = 1;
              for (int i = ndim - 1; i >= 0; --i) {
                XGRAMMAR_CHECK(t->shape[i] > 0 && (!t->strides || t->strides[i] == stride));
                stride *= t->shape[i];
              }
              XGRAMMAR_CHECK(t->data != nullptr);
              return t;
            };
            auto* inputs = tensor(inputs_value, 2, kDLInt, 0);
            auto* masks = tensor(masks_value, 3, kDLInt, 32);
            auto* temperatures = tensor(temps_value, 2, kDLFloat, 32);
            const int64_t batch = masks->shape[0], queries = masks->shape[1];
            const int64_t width = masks->shape[2];
            XGRAMMAR_CHECK(inputs->shape[0] == batch && inputs->shape[1] == queries
                && temperatures->shape[0] == batch && temperatures->shape[1] == queries);
            auto validate_words = [width](const auto& w, const auto& k) {
              XGRAMMAR_CHECK(w.size() == k.size());
              for (int64_t j = 0; j < static_cast<int64_t>(w.size()); ++j)
                XGRAMMAR_CHECK(w[j] >= 0 && w[j] < width
                    && k[j] >= INT32_MIN && k[j] <= INT32_MAX);
            };
            std::vector<GrammarMatcher> matchers;
            std::vector<std::vector<int32_t>> supplied(count);
            std::vector<std::exception_ptr> errors(count);
            std::unordered_set<int64_t> destinations;
            std::unordered_set<const void*> owners;
            matchers.reserve(count);
            bool parallel = count >= 8 && threads > 1;
            // Validate the entire batch before borrowing any parser.
            for (int64_t i = 0; i < count; ++i) {
              auto* object = matchers_ref[i].as<GrammarMatcherObj>();
              XGRAMMAR_CHECK(object && rows[i] >= 0 && rows[i] < batch
                  && destinations.insert(rows[i]).second && minimum[i] >= 0);
              auto matcher = object->value;
              parallel &= owners.insert(matcher.ImplPtr()).second;
              const auto rollback = matcher.GetMaxRollbackTokens();
              XGRAMMAR_CHECK(rollback < 0 || rollback >= queries - 1);
              matchers.push_back(std::move(matcher));
              validate_words(words[i], keeps[i]);
              validate_words(min_words[i], min_keeps[i]);
              ffi::AnyView provided = supplied_values[i];
              if (provided != nullptr) {
                auto* value = tensor(provided, 1, kDLInt, 32);
                XGRAMMAR_CHECK(value->shape[0] == width);
                auto* data = static_cast<int32_t*>(value->data);
                supplied[i].assign(data, data + width);
              }
              for (int64_t q = 1; q < queries; ++q) {
                const int64_t offset = rows[i] * queries + q;
                const int64_t token = inputs->dtype.bits == 32
                    ? static_cast<int32_t*>(inputs->data)[offset]
                    : static_cast<int64_t*>(inputs->data)[offset];
                XGRAMMAR_CHECK(token >= 0 && token < width * 32 && token <= INT32_MAX);
              }
            }
            auto run = [&](int64_t i) {
              try {
                const int64_t row = rows[i];
                int64_t shape[] = {queries, width};
                DLTensor view = *masks;
                view.data = static_cast<int32_t*>(masks->data) + row * queries * width;
                view.ndim = 2; view.shape = shape; view.strides = nullptr;
                auto* temps = static_cast<float*>(temperatures->data) + row * queries;
                auto* provided = supplied[i].empty() ? nullptr : supplied[i].data();
                if (inputs->dtype.bits == 32) {
                  TraverseFilteredChain(matchers[i],
                      static_cast<int32_t*>(inputs->data) + row * queries,
                      &view, temps, provided, words[i], keeps[i], minimum[i],
                      min_words[i], min_keeps[i]);
                } else {
                  TraverseFilteredChain(matchers[i],
                      static_cast<int64_t*>(inputs->data) + row * queries,
                      &view, temps, provided, words[i], keeps[i], minimum[i],
                      min_words[i], min_keeps[i]);
                }
              } catch (...) { errors[i] = std::current_exception(); }
            };
            if (parallel) {
              static ThreadPool pool(8);
              const int64_t chunks = std::min(threads, count);
              std::vector<std::shared_future<void>> tasks;
              tasks.reserve(chunks);
              std::exception_ptr dispatch_error;
              try {
                for (int64_t chunk = 0; chunk < chunks; ++chunk)
                  tasks.push_back(pool.Submit([&, chunk] {
                    for (int64_t i = count * chunk / chunks;
                         i < count * (chunk + 1) / chunks; ++i) run(i);
                  }));
              } catch (...) { dispatch_error = std::current_exception(); }
              // Every submitted task rejoins before an error can escape.
              for (auto& task : tasks) {
                try { task.get(); }
                catch (...) { if (!dispatch_error) dispatch_error = std::current_exception(); }
              }
              if (dispatch_error) std::rethrow_exception(dispatch_error);
            } else {
              for (int64_t i = 0; i < count; ++i) {
                run(i);
                if (errors[i]) break;
              }
            }
            for (const auto& error : errors) if (error) std::rethrow_exception(error);
          }
      )
      .def_static(
          "batch_accept_token_blocks",
          [](ffi::Array<O> matchers_ref, ffi::Array<int64_t> tokens,
             ffi::Array<int64_t> lengths) -> int64_t {
            // Commit in row/token order, with the same AcceptToken semantics as
            // scalar commits. Return the accepted prefix on failure; later rows
            // remain untouched. A scalar result avoids one FFI read per token.
            XGRAMMAR_CHECK(matchers_ref.size() == lengths.size());
            int64_t total = 0;
            for (int64_t i = 0; i < static_cast<int64_t>(lengths.size()); ++i) {
              XGRAMMAR_CHECK(lengths[i] >= 0 && lengths[i] <= static_cast<int64_t>(tokens.size()) - total);
              XGRAMMAR_CHECK(matchers_ref[i].as<GrammarMatcherObj>() != nullptr);
              total += lengths[i];
            }
            XGRAMMAR_CHECK(total == static_cast<int64_t>(tokens.size()));
            for (int64_t token : tokens) {
              XGRAMMAR_CHECK(token >= INT32_MIN && token <= INT32_MAX);
            }
            int64_t accepted = 0;
            for (int64_t i = 0; i < static_cast<int64_t>(lengths.size()); ++i) {
              auto matcher = matchers_ref[i].as<GrammarMatcherObj>()->value;
              for (int64_t j = 0; j < lengths[i]; ++j) {
                if (!matcher.AcceptToken(static_cast<int32_t>(tokens[accepted]))) {
                  return accepted;
                }
                ++accepted;
              }
            }
            return accepted;
          }
      )
      .def_static(
          "batch_accept_token",
          [](ffi::Array<O> matchers_ref, ffi::Array<int64_t> token_ids, bool debug_print) {
            std::vector<GrammarMatcher> matchers;
            matchers.reserve(matchers_ref.size());
            for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
              matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
            }
            std::vector<int32_t> token_id_vector;
            token_id_vector.reserve(token_ids.size());
            for (int64_t i = 0; i < static_cast<int64_t>(token_ids.size()); ++i) {
              token_id_vector.push_back(static_cast<int32_t>(token_ids[i]));
            }
            std::vector<uint8_t> acceptance_results =
                BatchGrammarMatcher::BatchAcceptToken(&matchers, token_id_vector, debug_print);
            ffi::Array<int64_t> acceptance_results_array;
            for (uint8_t acceptance_flag : acceptance_results)
              acceptance_results_array.push_back(static_cast<int64_t>(acceptance_flag));
            return acceptance_results_array;
          }
      )
      .def_static("batch_rollback", [](ffi::Array<O> matchers_ref, ffi::Array<int64_t> num_tokens) {
        std::vector<GrammarMatcher> matchers;
        matchers.reserve(matchers_ref.size());
        for (int64_t i = 0; i < static_cast<int64_t>(matchers_ref.size()); ++i) {
          matchers.push_back(matchers_ref[i].as<GrammarMatcherObj>()->value);
        }
        std::vector<int> num_tokens_vector;
        num_tokens_vector.reserve(num_tokens.size());
        for (int64_t i = 0; i < static_cast<int64_t>(num_tokens.size()); ++i) {
          num_tokens_vector.push_back(static_cast<int>(num_tokens[i]));
        }
        BatchGrammarMatcher::BatchRollback(&matchers, num_tokens_vector);
      });

  // GrammarMatcher: init(compiled_grammar, override_stop_tokens_opt, terminate_without_stop,
  // max_rollback_tokens, default_temperature)
  refl::ObjectDef<GrammarMatcherObj>()
      .def(refl::init<O, ffi::AnyView, bool, int64_t, std::optional<float>>())
      .def(
          "accept_token",
          [](GrammarMatcherObj* o, int64_t token_id, bool debug_print) {
            return o->value.AcceptToken(static_cast<int32_t>(token_id), debug_print);
          }
      )
      .def(
          "accept_string",
          [](GrammarMatcherObj* o, ffi::Any input_str_bytes_union, bool debug_print) {
            ffi::AnyView view = input_str_bytes_union;
            if (view.as<ffi::Bytes>()) {
              return o->value.AcceptString(BytesToString(view.cast<ffi::Bytes>()), debug_print);
            } else if (view.as<ffi::String>()) {
              return o->value.AcceptString(view.cast<ffi::String>(), debug_print);
            } else {
              TVM_FFI_THROW(RuntimeError) << "Unsupported type in accept_string";
              XGRAMMAR_UNREACHABLE();
            }
          }
      )
      .def(
          "fill_next_token_bitmask",
          [](GrammarMatcherObj* o, ffi::AnyView token_bitmask, int64_t index, bool debug_print) {
            DLTensor* dlt = token_bitmask.cast<DLTensor*>();
            return o->value.FillNextTokenBitmask(dlt, static_cast<int>(index), debug_print);
          }
      )
      .def(
          "traverse_filtered_chain",
          [](GrammarMatcherObj* o, ffi::AnyView inputs_value, ffi::AnyView masks_value,
             ffi::AnyView temperatures_value, ffi::AnyView supplied_value,
             ffi::Array<int64_t> words, ffi::Array<int64_t> keeps, int64_t minimum,
             ffi::Array<int64_t> min_words, ffi::Array<int64_t> min_keeps) {
            auto tensor = [](ffi::AnyView value, int ndim, int code, int bits) {
              auto* t = value.cast<DLTensor*>();
              XGRAMMAR_CHECK(t && t->device.device_type == kDLCPU && t->ndim == ndim
                             && t->dtype.code == code && t->dtype.bits == bits
                             && t->dtype.lanes == 1 && t->byte_offset == 0);
              int64_t stride = 1;
              for (int i = ndim - 1; i >= 0; --i) {
                XGRAMMAR_CHECK(t->shape[i] > 0 && (!t->strides || t->strides[i] == stride));
                stride *= t->shape[i];
              }
              XGRAMMAR_CHECK(t->data != nullptr);
              return t;
            };
            auto* inputs = tensor(inputs_value, 1, kDLInt, 64);
            auto* masks = tensor(masks_value, 2, kDLInt, 32);
            auto* temperatures = tensor(temperatures_value, 1, kDLFloat, 32);
            const int64_t queries = inputs->shape[0], width = masks->shape[1];
            XGRAMMAR_CHECK(masks->shape[0] == queries && temperatures->shape[0] == queries);
            const int64_t* tokens = static_cast<int64_t*>(inputs->data);
            auto* temps = static_cast<float*>(temperatures->data);
            DLTensor* supplied = supplied_value == nullptr ? nullptr
                : tensor(supplied_value, 1, kDLInt, 32);
            XGRAMMAR_CHECK(!supplied || supplied->shape[0] == width);
            auto validate_words = [width](auto w, auto k) {
              XGRAMMAR_CHECK(w.size() == k.size());
              for (int64_t i = 0; i < static_cast<int64_t>(w.size()); ++i) {
                XGRAMMAR_CHECK(w[i] >= 0 && w[i] < width
                               && k[i] >= INT32_MIN && k[i] <= INT32_MAX);
              }
            };
            validate_words(words, keeps); validate_words(min_words, min_keeps);
            for (int64_t i = 1; i < queries; ++i) {
              XGRAMMAR_CHECK(tokens[i] >= 0 && tokens[i] < width * 32
                             && tokens[i] <= INT32_MAX);
            }
            TraverseFilteredChain(o->value, tokens, masks, temps,
                supplied ? static_cast<int32_t*>(supplied->data) : nullptr,
                words, keeps, minimum, min_words, min_keeps);
          }
      )
      .def(
          "traverse_draft_tree",
          [](GrammarMatcherObj* o,
             ffi::AnyView retrieve_next_token,
             ffi::AnyView retrieve_next_sibling,
             ffi::AnyView draft_tokens,
             ffi::AnyView token_bitmask,
             double time_threshold,
             ffi::AnyView temperatures) {
            DLTensor* retrieve_next_token_ptr = retrieve_next_token.cast<DLTensor*>();
            DLTensor* retrieve_next_sibling_ptr = retrieve_next_sibling.cast<DLTensor*>();
            DLTensor* draft_tokens_ptr = draft_tokens.cast<DLTensor*>();
            DLTensor* token_bitmask_ptr = token_bitmask.cast<DLTensor*>();
            DLTensor* temperatures_ptr =
                temperatures == nullptr ? nullptr : temperatures.cast<DLTensor*>();
            return o->value.TraverseDraftTree(
                retrieve_next_token_ptr,
                retrieve_next_sibling_ptr,
                draft_tokens_ptr,
                token_bitmask_ptr,
                time_threshold,
                temperatures_ptr
            );
          }
      )
      .def(
          "find_jump_forward_string",
          [](GrammarMatcherObj* o) { return ffi::String(o->value.FindJumpForwardString()); }
      )
      .def(
          "rollback",
          [](GrammarMatcherObj* o, int64_t num_tokens) {
            o->value.Rollback(static_cast<int>(num_tokens));
          }
      )
      .def(
          "get_captures",
          [](const GrammarMatcherObj* o, bool deduplicate) {
            auto captures = o->value.GetCaptures(deduplicate);
            ffi::Array<ffi::Any> result;
            for (const auto& [name, value] : captures) {
              ffi::Array<ffi::Any> pair;
              pair.push_back(ffi::String(name));
              pair.push_back(ffi::Bytes(value));
              result.push_back(pair);
            }
            return result;
          }
      )
      .def(
          "fork",
          [](GrammarMatcherObj* o) {
            return ffi::ObjectRef(ffi::make_object<GrammarMatcherObj>(o->value.Fork()));
          }
      )
      .def("is_terminated", [](const GrammarMatcherObj* o) { return o->value.IsTerminated(); })
      .def("is_completed", [](const GrammarMatcherObj* o) { return o->value.IsCompleted(); })
      .def("reset", [](GrammarMatcherObj* o) { o->value.Reset(); })
      .def(
          "max_rollback_tokens",
          [](const GrammarMatcherObj* o) {
            return static_cast<int64_t>(o->value.GetMaxRollbackTokens());
          }
      )
      .def("temperature", [](const GrammarMatcherObj* o) { return o->value.GetTemperature(); })
      .def(
          "stop_token_ids",
          [](const GrammarMatcherObj* o) {
            const auto& stop_token_ids = o->value.GetStopTokenIds();
            ffi::Array<int64_t> stop_token_ids_array;
            for (int token_id : stop_token_ids)
              stop_token_ids_array.push_back(static_cast<int64_t>(token_id));
            return stop_token_ids_array;
          }
      )
      .def("_debug_print_internal_state", [](const GrammarMatcherObj* o) {
        return ffi::String(o->value._DebugPrintInternalState());
      });

  // ----- Global functions: testing, kernels, config, exceptions -----
  refl::GlobalDef()
      .def(
          "xgrammar.tvm_ffi_binding.testing._json_schema_to_ebnf",
          [](ffi::String schema,
             bool any_whitespace,
             ffi::AnyView indent,
             ffi::AnyView separators,
             bool strict_mode,
             ffi::AnyView max_whitespace_cnt,
             ffi::String json_format,
             bool any_order) {
            auto format = JSONFormatFromString(json_format);
            if (!format.has_value()) {
              TVM_FFI_THROW(RuntimeError) << "Invalid json_format: " << std::string(json_format);
            }
            return ffi::String(JSONSchemaToEBNF(
                schema,
                any_whitespace,
                OptionalIntFromView(indent),
                OptionalSeparatorsFromView(separators),
                strict_mode,
                OptionalIntFromView(max_whitespace_cnt),
                *format,
                any_order
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._regex_to_ebnf",
          [](ffi::String regex, ffi::AnyView with_rule_name_opt) {
            bool with_rule_name = OptionalBoolFromView(with_rule_name_opt).value_or(true);
            return ffi::String(RegexToEBNF(regex, with_rule_name));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._ebnf_to_grammar_no_normalization",
          [](ffi::String ebnf_str, ffi::String root_rule_name) {
            Grammar grammar = _EBNFToGrammarNoNormalization(ebnf_str, root_rule_name);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._get_masked_tokens_from_bitmask",
          [](int64_t token_bitmask_ptr, ffi::Array<int64_t> shape, int64_t vocab_size, int64_t index
          ) {
            std::vector<int64_t> shape_vector;
            for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i)
              shape_vector.push_back(shape[i]);
            std::vector<int> masked_tokens = Testing_DebugGetMaskedTokensFromBitmask(
                static_cast<intptr_t>(token_bitmask_ptr),
                shape_vector,
                static_cast<int32_t>(vocab_size),
                static_cast<int32_t>(index)
            );
            ffi::Array<int64_t> masked_tokens_array;
            for (int token_id : masked_tokens)
              masked_tokens_array.push_back(static_cast<int64_t>(token_id));
            return masked_tokens_array;
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._is_single_token_bitmask",
          [](int64_t token_bitmask_ptr, ffi::Array<int64_t> shape, int64_t vocab_size, int64_t index
          ) {
            std::vector<int64_t> shape_vector;
            for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i)
              shape_vector.push_back(shape[i]);
            auto single_token_result = Testing_IsSingleTokenBitmask(
                static_cast<intptr_t>(token_bitmask_ptr),
                shape_vector,
                static_cast<int32_t>(vocab_size),
                static_cast<int32_t>(index)
            );
            return ffi::Array<int64_t>{
                static_cast<int64_t>(single_token_result.first),
                static_cast<int64_t>(single_token_result.second)
            };
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._get_allow_empty_rule_ids",
          [](O compiled_grammar_ref) {
            const auto& compiled_grammar = compiled_grammar_ref.as<CompiledGrammarObj>()->value;
            std::vector<int32_t> allow_empty_rule_ids = GetAllowEmptyRuleIds(compiled_grammar);
            ffi::Array<int64_t> allow_empty_rule_ids_array;
            for (int32_t rule_id : allow_empty_rule_ids)
              allow_empty_rule_ids_array.push_back(static_cast<int64_t>(rule_id));
            return allow_empty_rule_ids_array;
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._generate_range_regex",
          [](ffi::AnyView start, ffi::AnyView end) {
            std::optional<int64_t> start_value = OptionalIntFromView(start);
            std::optional<int64_t> end_value = OptionalIntFromView(end);
            std::string regex_string = GenerateRangeRegex(start_value, end_value);
            regex_string.erase(
                std::remove(regex_string.begin(), regex_string.end(), '\0'), regex_string.end()
            );
            return ffi::String(regex_string);
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._generate_float_regex",
          [](ffi::AnyView start, ffi::AnyView end, bool exclusive_start, bool exclusive_end) {
            std::optional<double> start_value =
                start == nullptr ? std::nullopt : std::make_optional(start.cast<double>());
            std::optional<double> end_value =
                end == nullptr ? std::nullopt : std::make_optional(end.cast<double>());
            std::string regex_string =
                GenerateFloatRangeRegex(start_value, end_value, exclusive_start, exclusive_end);
            regex_string.erase(
                std::remove(regex_string.begin(), regex_string.end(), '\0'), regex_string.end()
            );
            return ffi::String(regex_string);
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._print_grammar_fsms",
          [](O grammar_ref) {
            return ffi::String(_PrintGrammarFSMs(grammar_ref.as<GrammarObj>()->value));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._is_rule_fsm_accept_string",
          [](O grammar_ref, int64_t rule_id, ffi::String input) {
            const auto& grammar = grammar_ref.as<GrammarObj>()->value;
            XGRAMMAR_CHECK(
                rule_id >= 0 && rule_id < static_cast<int64_t>(grammar->per_rule_fsms.size())
            ) << "Rule id is out of range or the grammar has no per-rule FSMs: "
              << rule_id;
            const auto& rule_fsm = grammar->per_rule_fsms[rule_id];
            XGRAMMAR_CHECK(rule_fsm.has_value())
                << "Rule " << rule_id << " does not have a finite-state machine";
            return rule_fsm->GetFsm().AcceptString(input);
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.structure_normalizer",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                StructureNormalizer::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.byte_string_fuser",
          [](O grammar_ref) {
            // The pass rewrites in place, so copy first to leave the input grammar unaffected.
            const Grammar& input = grammar_ref.as<GrammarObj>()->value;
            Grammar grammar(std::make_shared<Grammar::Impl>(*input.operator->()));
            ByteStringFuser::Apply(&grammar);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(grammar));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.rule_inliner",
          [](O grammar_ref) {
            // The pass rewrites in place, so copy first to leave the input grammar unaffected.
            const Grammar& input = grammar_ref.as<GrammarObj>()->value;
            Grammar grammar(std::make_shared<Grammar::Impl>(*input.operator->()));
            RuleInliner::Apply(&grammar);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(grammar));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.dead_code_eliminator",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                DeadCodeEliminator::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.lookahead_assertion_analyzer",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                LookaheadAssertionAnalyzer::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.grammar_optimizer",
          [](O grammar_ref) {
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(
                GrammarOptimizer::Apply(grammar_ref.as<GrammarObj>()->value)
            ));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.fsm_builder",
          [](O grammar_ref) {
            Grammar grammar = grammar_ref.as<GrammarObj>()->value;
            GrammarFSMBuilder::Apply(&grammar);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing.grammar_functor.repetition_normalizer",
          [](O grammar_ref) {
            Grammar grammar = grammar_ref.as<GrammarObj>()->value;
            RepetitionNormalizer::Apply(&grammar);
            return ffi::ObjectRef(ffi::make_object<GrammarObj>(std::move(grammar)));
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.kernels.apply_token_bitmask_inplace_cpu",
          [](int64_t logits_ptr,
             ffi::Array<int64_t> logits_shape,
             ffi::Array<int64_t> logits_strides,
             int64_t bitmask_ptr,
             ffi::Array<int64_t> bitmask_shape,
             ffi::Array<int64_t> bitmask_strides,
             int64_t vocab_size,
             ffi::AnyView indices,
             ffi::String logit_type) {
            Kernels_ApplyTokenBitmaskInplaceCPU(
                static_cast<intptr_t>(logits_ptr),
                {logits_shape[0], logits_shape[1]},
                {logits_strides[0], logits_strides[1]},
                static_cast<intptr_t>(bitmask_ptr),
                {bitmask_shape[0], bitmask_shape[1]},
                {bitmask_strides[0], bitmask_strides[1]},
                static_cast<int>(vocab_size),
                OptionalIntVectorFromView(indices),
                logit_type
            );
          }
      )
      .def(
          "xgrammar.tvm_ffi_binding.config.set_max_recursion_depth",
          [](int64_t depth) { SetMaxRecursionDepth(static_cast<int>(depth)); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.config.get_max_recursion_depth",
          []() { return static_cast<int64_t>(GetMaxRecursionDepth()); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.config.get_serialization_version",
          []() { return ffi::String(GetSerializationVersion()); }
      )
      .def(
          "xgrammar.tvm_ffi_binding.testing._traverse_draft_tree",
          [](ffi::AnyView retrieve_next_token,
             ffi::AnyView retrieve_next_sibling,
             ffi::AnyView draft_tokens,
             O matcher_ref,
             ffi::AnyView bitmask,
             ffi::AnyView time_threshold_opt) {
            XGRAMMAR_FFI_TRY_BEGIN();
            DLTensor* retrieve_next_token_ptr = retrieve_next_token.cast<DLTensor*>();
            DLTensor* retrieve_next_sibling_ptr = retrieve_next_sibling.cast<DLTensor*>();
            DLTensor* draft_tokens_ptr = draft_tokens.cast<DLTensor*>();
            DLTensor* bitmask_ptr = bitmask.cast<DLTensor*>();
            double time_threshold =
                time_threshold_opt == nullptr ? -1.0 : time_threshold_opt.cast<double>();
            GrammarMatcher& matcher =
                const_cast<GrammarMatcher&>(matcher_ref.as<GrammarMatcherObj>()->value);
            return matcher.TraverseDraftTree(
                retrieve_next_token_ptr,
                retrieve_next_sibling_ptr,
                draft_tokens_ptr,
                bitmask_ptr,
                time_threshold
            );
            XGRAMMAR_FFI_TRY_END();
          }
      );
}

}  // namespace xgrammar
