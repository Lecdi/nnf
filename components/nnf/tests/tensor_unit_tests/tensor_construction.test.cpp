#include <catch2/catch_test_macros.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/vector.hpp>

using namespace nnf;

TEST_CASE("Tensor construction gives contiguous tensor with correct rank, dims and num elems", "[tensor]")
{
	SECTION("Empty scalar");
	Tensor a{ Vector<usize>{} };
	REQUIRE(a.is_contiguous());
	REQUIRE(a.rank() == 0);
	REQUIRE(a.view_dims().size() == 0);
	REQUIRE(a.num_elems() == 1);

	SECTION("Empty rank 3 tensor");
	Tensor b{ {1, 5, 8} };
	REQUIRE(b.is_contiguous());
	REQUIRE(b.rank() == 3);
	REQUIRE(b.view_dims().size() == 3);
	REQUIRE(b.view_dims()[0] == 1);
	REQUIRE(b.view_dims()[1] == 5);
	REQUIRE(b.view_dims()[2] == 8);

	SECTION("Zero scalar");
	auto c = Tensor::zeroes(Vector<usize>{});
	REQUIRE(c.is_contiguous());
	REQUIRE(c.rank() == 0);
	REQUIRE(c.view_dims().size() == 0);
	REQUIRE(c.num_elems() == 1);

	SECTION("Zero rank 3 tensor");
	auto d = Tensor::zeroes({ 1, 5, 8 });
	REQUIRE(d.is_contiguous());
	REQUIRE(d.rank() == 3);
	REQUIRE(d.view_dims().size() == 3);
	REQUIRE(d.view_dims()[0] == 1);
	REQUIRE(d.view_dims()[1] == 5);
	REQUIRE(d.view_dims()[2] == 8);

	SECTION("Normal scalar");
	auto e = Tensor::normals(Vector<usize>{}, 0, 1);
	REQUIRE(e.is_contiguous());
	REQUIRE(e.rank() == 0);
	REQUIRE(e.view_dims().size() == 0);
	REQUIRE(e.num_elems() == 1);

	SECTION("Normal rank 3 tensor");
	auto f = Tensor::normals({ 1, 5, 8 }, -3, 4);
	REQUIRE(f.is_contiguous());
	REQUIRE(f.rank() == 3);
	REQUIRE(f.view_dims().size() == 3);
	REQUIRE(f.view_dims()[0] == 1);
	REQUIRE(f.view_dims()[1] == 5);
	REQUIRE(f.view_dims()[2] == 8);

	SECTION("From data scalar");
	auto g = Tensor::from_data(Vector<usize>{}, { 3.f });
	REQUIRE(g.is_contiguous());
	REQUIRE(g.rank() == 0);
	REQUIRE(g.view_dims().size() == 0);
	REQUIRE(g.num_elems() == 1);

	SECTION("From data rank 3 tensor");
	auto h = Tensor::from_data({ 1, 2, 3 }, { 0, 1, 2, 3, 4, 5 });
	REQUIRE(h.is_contiguous());
	REQUIRE(h.rank() == 3);
	REQUIRE(h.view_dims().size() == 3);
	REQUIRE(h.view_dims()[0] == 1);
	REQUIRE(h.view_dims()[1] == 2);
	REQUIRE(h.view_dims()[2] == 3);
}

TEST_CASE("Empty tensor construction gives empty tensor", "[tensor]")
{
	SECTION("Empty scalar");
	Tensor a{ Vector<usize>{} };
	REQUIRE(a.is_empty());

	SECTION("Empty rank 3 tensor");
	Tensor b{ {1, 5, 8} };
	REQUIRE(b.is_empty());
}

TEST_CASE("Construction of tensor with 0 dimension throws", "[tensor]")
{
	SECTION("Empty");
	REQUIRE_THROWS(Tensor({ 0 }));
	REQUIRE_THROWS(Tensor({ 1, 0 }));
	REQUIRE_THROWS(Tensor({ 0, 1, 3 }));
	REQUIRE_THROWS(Tensor({ 4, 0, 3 }));

	SECTION("Zeroes");
	REQUIRE_THROWS(Tensor::zeroes({ 0 }));
	REQUIRE_THROWS(Tensor::zeroes({ 1, 0 }));
	REQUIRE_THROWS(Tensor::zeroes({ 0, 1, 3 }));
	REQUIRE_THROWS(Tensor::zeroes({ 4, 0, 3 }));

	SECTION("Normals");
	REQUIRE_THROWS(Tensor::normals({ 0 }, 1, 3));
	REQUIRE_THROWS(Tensor::normals({ 1, 0 }, 0, 1));
	REQUIRE_THROWS(Tensor::normals({ 0, 1, 3 }, 6, 7));
	REQUIRE_THROWS(Tensor::normals({ 4, 0, 3 }, 0, 1));

	SECTION("From data");
	REQUIRE_THROWS(Tensor::from_data({ 0 }, { {} }));
	REQUIRE_THROWS(Tensor::from_data({ 1, 0 }, { {} }));
	REQUIRE_THROWS(Tensor::from_data({ 0, 1, 3 }, { {2, 3} }));
	REQUIRE_THROWS(Tensor::from_data({ 4, 0, 3 }, { {1, 4, 5, 6} }));
}

TEST_CASE("Construction of tensor from data with incorrect number of elements throws", "[tensor]")
{
	SECTION("Scalar");
	REQUIRE_THROWS(Tensor::from_data({ {} }, { {} }));
	REQUIRE_THROWS(Tensor::from_data({ {} }, { 3.f, 4.f, 5.f }));

	SECTION("Rank 3 tensor");
	REQUIRE_THROWS(Tensor::from_data({ 1, 2, 3 }, { 3, 4, 5, 6, 7 }));
	REQUIRE_THROWS(Tensor::from_data({ 1, 2, 3 }, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f }));
}
