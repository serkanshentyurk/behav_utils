# API

One page per sub-package, documenting the names in its `__all__`. Every public name has one home; the most
used can also be imported from `behav_utils` itself, and the [public API index](../public_api.md) marks those
*top level*. Module paths below a sub-package, and names starting with `_`, are private and can change in any
release.

```{toctree}
:maxdepth: 1

config
tasks
data
stats
readouts
models
analysis
plotting
```
