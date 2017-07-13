<h1 align="center">
  &chi;
</h1>

<p align="center">
  `cai` (&chi;) Dependency-free, simple and extensible Deep Learning library, written in C.
</p>

<h2 align="center">Building</h2>

<p align="center">
  Build the `cai` library
</p>

```sh
make
```

<h2 align="center">Linking</h2>

<p align="center">
  Linking and using the `cai` library
</p>

```sh
LD_FLAGS=-lcai -I$(CAI)
```

<h2 align="center">Usage</h2>

<p align="center">
  <a href="https://github.com/wenkesj/cai/tree/master/xor">Full example here!</a>
</p>

```c
// All you'll ever need...
#include <cai/criterion.h>
#include <cai/matrix.h>
#include <cai/layer.h>
#include <cai/network.h>
```
