#pragma once

namespace nnf
{
    constexpr float sgn(float x)
	{
		return (x > 0) - (x < 0);
	}
    
    constexpr bool float_approx_eq(float lhs, float rhs, float epsilon = 1e-5f)
    {
    	return -epsilon < lhs - rhs && lhs - rhs < epsilon;
    }
}
