#include <catch2/catch_test_macros.hpp>
#include <nnf/tensor/tensor.hpp>

using namespace nnf;

TEST_CASE("Tensor addition and subtraction", "[tensor]")
{
	auto M = Tensor::from_data({ 5, 3 }, {
		 3,  0, -2,
		 0, -1,  0,
		 1,  0,  0,
		 2,  1,  1,
		-1,  0,  2
	});

	auto P = M.gather(0, { 1, 2, 4 });

	auto Q = Tensor::from_data({ 3, 3 }, {
		-1, -1,  3,
		 4,  2,  3,
		 0,  9,  0
	});

	auto N = Tensor::from_data({ 5, 3 }, {
		-1, -1,  3,
		 4,  2,  3,
		 0,  9,  0,
		 3,  1,  2,
		 4, -3, -5
	});

	REQUIRE(approx_eq(M + N, Tensor::from_data({ 5, 3 }, {
		2, -1,  1,
		4,  1,  3,
		1,  9,  0,
		5,  2,  3,
		3, -3, -3
	})));

	REQUIRE(approx_eq(M - N, Tensor::from_data({ 5, 3 }, {
		 4,  1, -5,
		-4, -3, -3,
		 1, -9,  0,
		-1,  0, -1,
		-5,  3,  7
	})));

	REQUIRE(approx_eq(P + Q, Tensor::from_data({ 3, 3 }, {
		-1, -2,  3,
		 5,  2,  3,
		-1,  9,  2
	})));

	REQUIRE(approx_eq(Q + P, Tensor::from_data({ 3, 3 }, {
		-1, -2,  3,
		 5,  2,  3,
		-1,  9,  2
	})));

	REQUIRE(approx_eq(P - Q, Tensor::from_data({ 3, 3 }, {
		 1,  0, -3,
		-3, -2, -3,
		-1, -9,  2
	})));

	REQUIRE(approx_eq(Q - P, Tensor::from_data({ 3, 3 }, {
		-1,  0,  3,
		 3,  2,  3,
		 1,  9, -2
	})));

	REQUIRE_THROWS(M + P);
	REQUIRE_THROWS(M + Q);
	REQUIRE_THROWS(N + P);
	REQUIRE_THROWS(N + Q);
	REQUIRE_THROWS(P + M);
	REQUIRE_THROWS(Q + M);
	REQUIRE_THROWS(P + N);
	REQUIRE_THROWS(Q + N);
	REQUIRE_THROWS(M - P);
	REQUIRE_THROWS(M - Q);
	REQUIRE_THROWS(N - P);
	REQUIRE_THROWS(N - Q);
	REQUIRE_THROWS(P - M);
	REQUIRE_THROWS(Q - M);
	REQUIRE_THROWS(P - N);
	REQUIRE_THROWS(Q - N);
}

TEST_CASE("Tensor matvecmul", "[tensor]")
{
	auto M = Tensor::from_data({ 5, 3 }, {
		 3,  0, -2,
		 0, -1,  0,
		 1,  0,  0,
		 2,  1,  1,
		-1,  0,  2
	});

	auto u = Tensor::from_data({ 3 }, {
		 3,
		-4,
		 5
	});

	auto v = matvecmul(M, u);

	REQUIRE(approx_eq(v, Tensor::from_data({ 5 }, {
		-1,
		 4,
		 3,
		 7,
		 7
	})));
}

TEST_CASE("Tensor matmatmul", "[tensor]")
{
	auto P = Tensor::from_data({ 5, 3 }, {
		 3,  0, -2,
		 0, -1,  0,
		 1,  0,  0,
		 2,  1,  1,
		-1,  0,  2
	});

	auto Q = Tensor::from_data({ 3, 2 }, {
		 3, -4,
		 5, -6,
		-7,  8
	});

	auto R = matmatmul(P, Q);

	REQUIRE(approx_eq(R, Tensor::from_data({ 5, 2 }, {
		23, -28,
		-5,   6,
		 3,  -4,
		 4,  -6,
	   -17,  20
	})));
}

TEST_CASE("Tensor matmul", "[tensor]")
{
	auto M = Tensor::from_data({ 5, 3 }, {
		 3,  0, -2,
		 0, -1,  0,
		 1,  0,  0,
		 2,  1,  1,
		-1,  0,  2
	});

	auto u = Tensor::from_data({ 3 }, {
		 3,
		-4,
		 5
	});

	auto v = matmul(M, u);

	REQUIRE(approx_eq(v, Tensor::from_data({ 5 }, {
		-1,
		 4,
		 3,
		 7,
		 7
	})));

	Tensor P = M;

	auto Q = Tensor::from_data({ 3, 2 }, {
		 3, -4,
		 5, -6,
		-7,  8
	});

	auto R = matmul(P, Q);

	REQUIRE(approx_eq(R, Tensor::from_data({ 5, 2 }, {
		23, -28,
		-5,   6,
		 3,  -4,
		 4,  -6,
	   -17,  20
	})));
}

TEST_CASE("Tensor fold2d", "[tensor]")
{
	auto T = Tensor::from_data({ 15, 9 }, {
		 -4,  2,  3, -1,  0, -4,  2, -3,  0,
		 -2,  4,  5,  0,  1,  4, -2,  0,  1,
		  3,  5, -2,  0,  2, -5,  0,  0,  0,
		  0, -1, -5, -3,  3,  0, -4,  0,  4,
		  0,  0,  6, -2, -4, -8,  2,  1, -3,
		  5, -3, -7,  0,  0,  3,  1,  2, -3,
		 -9,  0,  2,  1,  0,  2,  2, -3,  0,
		  3,  2, -4,  2,  0, -1,  3, -4, -5,
		  0,  2,  0,  3,  0,  1,  0,  1,  1,
		 -2,  0,  0,  0,  5,  3, -4, -5,  2,
		  1, -2,  3, -4,  6, -5, -6,  7, -7,
		  0,  3,  4,  2,  7,  0,  2,  0,  4,
		 -4, -5, -2,  3, -4, -3,  3, -6,  3,
		  0,  0,  1, -2,  0, -4,  0,  0, -2,
		 -1,  0,  0, -1, -3,  1, -1, -3,  1,
	});

	auto F = T.fold2d(11, 8, 3, 3, 2, 2);

	REQUIRE(approx_eq(F, Tensor::from_data({ 11, 8 }, {
	 	 -4,  2,  1,  4,  8,  5, -2,  0,
	 	 -1,  0, -4,  1,  4,  2, -5,  0,
	 	  2, -4, -7,  0, 12, -3, -7,  0,
	 	 -3,  3, -2, -4, -8,  0,  3,  0,
	    -13,  0, 11,  3, -6,  4, -3,  0,
	 	  1,  0,  4,  0,  2,  0,  1,  0,
	 	  0, -3,  4, -6, -2,  4,  5,  0,
	 	  0,  5, -1,  6, -3,  7,  0,  0,
	 	 -8,-10, -6,  7, -5,  0,  4,  0,
	 	  3, -4, -5,  0, -5, -3,  1,  0,
	 	  3, -6,  3,  0, -3, -3,  1,  0
	})));
}
