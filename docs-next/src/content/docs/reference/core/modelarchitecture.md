---
title: ModelArchitecture
description: API reference for ModelArchitecture
---

# ModelArchitecture

```python
from modal_training_gym.common.models.base import ModelArchitecture
```

Transformer architecture parameters for a specific model.

## Model Dimensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_layers` | `int` |  | Number of transformer layers. Default `0`. |
| `hidden_size` | `int` |  | Hidden dimension size. Default `0`. |
| `ffn_hidden_size` | `int` |  | Feed-forward network intermediate size. Default `0`. |
| `vocab_size` | `int` |  | Vocabulary size. Default `0`. |

## Attention

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_attention_heads` | `int` |  | Number of attention heads. Default `0`. |
| `group_query_attention` | `bool` |  | Enable grouped-query attention (GQA). Default `True`. |
| `num_query_groups` | `int` |  | Number of KV head groups for GQA. Default `0`. |
| `kv_channels` | `int` |  | Per-head key/value channel dimension. Default `0`. |

## Normalization and Activation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalization` | `str` |  | Layer normalization type. Default `"RMSNorm"`. |
| `norm_epsilon` | `float` |  | Normalization epsilon. Default `1e-6`. |
| `swiglu` | `bool` |  | Use SwiGLU activation in FFN. Default `True`. |
| `disable_bias_linear` | `bool` |  | Disable bias in linear layers. Default `True`. |
| `qk_layernorm` | `bool` |  | Apply layer norm to query and key projections. Default `True`. |

## Position Encoding

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_rotary_position_embeddings` | `bool` |  | Use RoPE positional encoding. Default `True`. |
| `rotary_base` | `int` |  | Base frequency for RoPE. Default `10000`. |

## Methods

### `construct(_fields_set: 'set[str] | None' = None, **values: 'Any') -> 'Self'`

### `copy(self, *, include: 'AbstractSetIntStr | MappingIntStrAny | None' = None, exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None, update: 'Dict[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'`

Returns a copy of the model.

### `dict(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False) -> 'Dict[str, Any]'`

### `from_orm(obj: 'Any') -> 'Self'`

### `json(self, *, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool' = False, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, encoder: 'Callable[[Any], Any] | None' = PydanticUndefined, models_as_dict: 'bool' = PydanticUndefined, **dumps_kwargs: 'Any') -> 'str'`

### `model_construct(_fields_set: 'set[str] | None' = None, **values: 'Any') -> 'Self'`

Creates a new instance of the `Model` class with validated data.

### `model_copy(self, *, update: 'Mapping[str, Any] | None' = None, deep: 'bool' = False) -> 'Self'`

!!! abstract "Usage Documentation"

### `model_dump(self, *, mode: "Literal['json', 'python'] | str" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False, polymorphic_serialization: 'bool | None' = None) -> 'dict[str, Any]'`

!!! abstract "Usage Documentation"

### `model_dump_json(self, *, indent: 'int | None' = None, ensure_ascii: 'bool' = False, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, exclude_computed_fields: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False, polymorphic_serialization: 'bool | None' = None) -> 'str'`

!!! abstract "Usage Documentation"

### `model_json_schema(by_alias: 'bool' = True, ref_template: 'str' = '#/$defs/{model}', schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>, mode: 'JsonSchemaMode' = 'validation', *, union_format: "Literal['any_of', 'primitive_type_array']" = 'any_of') -> 'dict[str, Any]'`

Generates a JSON schema for a model class.

### `model_parametrized_name(params: 'tuple[type[Any], ...]') -> 'str'`

Compute the class name for parametrizations of generic classes.

### `model_post_init(self, context: 'Any', /) -> 'None'`

Override this method to perform additional initialization after `__init__` and `model_construct`.

### `model_rebuild(*, force: 'bool' = False, raise_errors: 'bool' = True, _parent_namespace_depth: 'int' = 2, _types_namespace: 'MappingNamespace | None' = None) -> 'bool | None'`

Try to rebuild the pydantic-core schema for the model.

### `model_validate(obj: 'Any', *, strict: 'bool | None' = None, extra: 'ExtraValues | None' = None, from_attributes: 'bool | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, by_name: 'bool | None' = None) -> 'Self'`

Validate a pydantic model instance.

### `model_validate_json(json_data: 'str | bytes | bytearray', *, strict: 'bool | None' = None, extra: 'ExtraValues | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, by_name: 'bool | None' = None) -> 'Self'`

!!! abstract "Usage Documentation"

### `model_validate_strings(obj: 'Any', *, strict: 'bool | None' = None, extra: 'ExtraValues | None' = None, context: 'Any | None' = None, by_alias: 'bool | None' = None, by_name: 'bool | None' = None) -> 'Self'`

Validate the given object with string data against the Pydantic model.

### `parse_file(path: 'str | Path', *, content_type: 'str | None' = None, encoding: 'str' = 'utf8', proto: 'DeprecatedParseProtocol | None' = None, allow_pickle: 'bool' = False) -> 'Self'`

### `parse_obj(obj: 'Any') -> 'Self'`

### `parse_raw(b: 'str | bytes', *, content_type: 'str | None' = None, encoding: 'str' = 'utf8', proto: 'DeprecatedParseProtocol | None' = None, allow_pickle: 'bool' = False) -> 'Self'`

### `schema(by_alias: 'bool' = True, ref_template: 'str' = '#/$defs/{model}') -> 'Dict[str, Any]'`

### `schema_json(*, by_alias: 'bool' = True, ref_template: 'str' = '#/$defs/{model}', **dumps_kwargs: 'Any') -> 'str'`

### `to_megatron_args(self) -> 'list[str]'`

Generate Megatron-LM CLI flags from this architecture spec.

### `update_forward_refs(**localns: 'Any') -> 'None'`

### `validate(value: 'Any') -> 'Self'`

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
