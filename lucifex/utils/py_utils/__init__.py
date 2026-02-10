from .deferred import defer, Writer
from .func_utils import (
    log_timing, optional_lru_cache, arity,
    MultipleDispatchTypeError, replicate_callable,
    filter_kwargs, canonicalize_args, MultipleDispatchTypeError,
)
from .str_utils import (
    str_indexed, str_scientifc, 
    str_plain, str_tex, StrSlice, as_slice, is_slice,
)
from .cls_utils import StrEnum, FloatEnum, classproperty, ToDoError