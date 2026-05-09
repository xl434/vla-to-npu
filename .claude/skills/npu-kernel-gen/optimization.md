# Approximation Technique Examples
Use `aie::select(x, y, condition)`for selecting methods: 
- x if condition is false
- y if condition is true

For condition in select, use: 
- `aie::gt(x, y)`, x > y
- `aie::lt(x, y)`, x < y
- `aie::ge(x, y)`, x >= y
- `aie::le(x, y)`, x <= y
- `aie::eq(x, y)`, x == y
- `aie::ne(x, y)`, x != y

## Periodic Functions
```c
// input restriction 0 < x < π/2 (for accuracy) 
template<unsigned vec_factor>
static inline aie::vector<float, vec_factor> mod_halfpi_vec(
  const aie::vector<float, vec_factor>& x,
  aie::mask<vec_factor>& neg_mask)
{
  using fvec_t = aie::vector<float, vec_factor>;

  const fvec_t inv_two_pi = aie::broadcast<float, vec_factor>(INV_TWO_PI);
  const fvec_t two_pi     = aie::broadcast<float, vec_factor>(TWO_PI_F);
  const fvec_t pi         = aie::broadcast<float, vec_factor>(PI_F);
  const fvec_t half_pi    = aie::broadcast<float, vec_factor>(HALF_PI_F);
  const fvec_t zero       = aie::broadcast<float, vec_factor>(0.0f);

  fvec_t q = aie::mul(x, inv_two_pi);
  auto n_i      = aie::to_fixed(q, 0);
  auto n_f      = aie::to_float(n_i, 0);
  q             = aie::negmul(n_f, two_pi);
  fvec_t r = aie::add(x, q);

  auto m_lt0   = aie::lt(r, zero);
  r = aie::select(r, aie::add(r, two_pi), m_lt0);
  auto m_ge2pi = aie::ge(r, two_pi);
  r = aie::select(r, aie::sub(r, two_pi), m_ge2pi);

  neg_mask = aie::ge(r, pi);
  r = aie::select(r, aie::sub(r, pi), neg_mask);

  auto m_gt_hpi = aie::gt(r, half_pi);
  r = aie::select(r, aie::sub(pi, r), m_gt_hpi);

  return r;
}

```

## Linear Approximation for edge cases

```c
    /* Original approximation method for sigmoid
    * Saturates to 0 for inputs < -7
    * Saturates to 1 for inputs > 7
    * Inaccurate for 4 < input < 7, use linear approximation in this range
    */
    fvec_t sigmoid = ...; 
    fvec_t linear  = linear_approx<float, vec_factor>(x);
    sigmoid = aie::select(sigmoid, linear, aie::gt(x, fpos4));
    sigmoid = aie::select(sigmoid, one,    aie::gt(x, fpos7));
    sigmoid = aie::select(sigmoid, fzero,  aie::lt(x, fneg7));
```

# Memory Overflow: reducing aie::vector type variable example

```c
    // Reusing xtemp
    // Taylor: x - x³/3! + x⁵/5! - x^7/7!
    fvec_t xtemp          = aie::mul(x, x);              // x^2
    fvec_t x3             = aie::mul(xtemp, x);          // x^3
    fvec_t x3div3fac      = aie::negmul(x3, cdiv3fac);   // -x^3/3!
    xtemp                 = aie::mul(x3, x);             // x^4
    fvec_t x5             = aie::mul(xtemp, x);          // x^5
    fvec_t x5div5fac      = aie::mul(x5, cdiv5fac);      // +x^5/5!
    xtemp                 = aie::mul(x5, x);             // x^6
    fvec_t x7             = aie::mul(xtemp, x);          // x^7
    fvec_t x7div7fac      = aie::negmul(x7, cdiv7fac);   // -x^7/7!

```