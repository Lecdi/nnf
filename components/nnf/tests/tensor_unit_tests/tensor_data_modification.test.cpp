#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/math.hpp>
#include <nnf/utils/vector.hpp>

using namespace nnf;

TEST_CASE("Tensor clear() produces an empty and detached tensor", "[tensor]")
{
	auto a = Tensor::zeroes({ 3, 4 });
	auto b = a;
	auto c = Tensor::zeroes({ 2 });

	b.clear();
	c.clear();

	REQUIRE(b.is_detached());
	REQUIRE(b.is_empty());
	REQUIRE(c.is_detached());
	REQUIRE(c.is_empty());
}

TEST_CASE("Tensor clear() does not clear attached tensors", "[tensor]")
{
	auto a = Tensor::zeroes({ 3, 4 });
	auto b = a;

	b.clear();

	REQUIRE(a.is_detached());
	REQUIRE(!a.is_empty());
}

TEST_CASE("Tensor clear() does not affect dimensions of tensor and reverts to contiguous tensor", "[tensor]")
{
	auto a = Tensor::zeroes({ 3, 4 });
	auto b = a.transposed();
	auto c = Tensor::zeroes({ 2 });
	REQUIRE(!b.is_contiguous());

	b.clear();
	c.clear();

	Vector<usize> b_dims_expected{ 4, 3 };
	Vector<usize> c_dims_expected{ 2 };
	auto b_dims = b.get_dims();
	auto c_dims = c.get_dims();

	REQUIRE(b_dims == b_dims_expected);
	REQUIRE(c_dims == c_dims_expected);
	REQUIRE(b.is_contiguous());
	REQUIRE(c.is_contiguous());
}

TEST_CASE("Modifying detached tensor does not affect previously attached tensors", "[tensor]")
{
	auto a = Tensor::zeroes({ 3, 4 });
	auto b = a;
	auto c = b.copy();
	auto d = b;
	d.detach();

	c.at({ { 0, 0 } }).set(1);
	d.at({ {0, 1} }).set(2);
	REQUIRE(c.at({ { 0, 0 } }) == 1);
	REQUIRE(d.at({ { 0, 1 } }) == 2);

	REQUIRE(a.at({ { 0, 0 } }) == 0);
	REQUIRE(b.at({ { 0, 0 } }) == 0);
	REQUIRE(a.at({ { 0, 1 } }) == 0);
	REQUIRE(b.at({ { 0, 1 } }) == 0);
}

TEST_CASE("Tensor require_contiguous() result is always contiguous when possible", "[tensor]")
{
	Tensor a{ Vector<usize>{} };
	auto b = a;
	auto c = Tensor::zeroes({ 3, 4 });
	auto d = c;
	auto e = c.reshaped({ 12, 1 });
	auto f = c;
	f.reshape_inplace({ 1, 12 });
	auto g = f.transposed();
	auto h = Tensor::zeroes({ 4, 5 }).transposed();

	a.require_contiguous();
	b.require_contiguous();
	c.require_contiguous();
	d.require_contiguous();
	e.require_contiguous();
	f.require_contiguous();
	h.require_contiguous();

	REQUIRE(a.is_contiguous());
	REQUIRE(b.is_contiguous());
	REQUIRE(c.is_contiguous());
	REQUIRE(d.is_contiguous());
	REQUIRE(e.is_contiguous());
	REQUIRE(f.is_contiguous());
	REQUIRE(h.is_contiguous());
}

TEST_CASE("Tensor require_contiguous() throws when not possible", "[tensor]")
{
	Tensor a{ {3, 4} };
	auto b = a.transposed();
	auto c = a;
	c.transpose_inplace();
	REQUIRE_THROWS(b.require_contiguous());
	REQUIRE_THROWS(c.require_contiguous());

	auto d = Tensor::from_data({ 1, 2, 3 }, { 1, 2, 3, 4, 5, 6 });
	auto e = d.permuted({ 1, 0, 2 });
	auto f = d;
	f.permute_inplace({ 2, 0, 1 });
	REQUIRE_THROWS(e.require_contiguous());
	REQUIRE_THROWS(f.require_contiguous());
}

TEST_CASE("Tensor set() results in equal tensors including for attached tensors", "[tensor]")
{
	SECTION("Setting non-empty matrix");

	auto a = Tensor::zeroes({ 3, 4 });
	auto b = a;
	auto c = b.copy();
	auto d = b;
	d.detach();

	auto f = Tensor::from_data({ 3, 4 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
	auto g = Tensor::from_data({ 3, 4 }, { 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7 });
	b.set(f);
	d.set(g);

	REQUIRE(a.at({ { 0, 0 } }) == 1);
	REQUIRE(a.at({ { 0, 1 } }) == 2);
	REQUIRE(a.at({ { 2, 1 } }) == 10);
	REQUIRE(b.at({ { 0, 0 } }) == 1);
	REQUIRE(b.at({ { 0, 1 } }) == 2);
	REQUIRE(b.at({ { 2, 1 } }) == 10);

	REQUIRE(c.at({ { 0, 0 } }) == 0);
	REQUIRE(c.at({ { 0, 3 } }) == 0);
	REQUIRE(c.at({ { 2, 1 } }) == 0);

	REQUIRE(d.at({ { 0, 1 } }) == 6);
	REQUIRE(d.at({ { 2, 3 } }) == 7);

	SECTION("Setting empty scalar");

	Tensor h{ Vector<usize>{} };
	auto i = Tensor::from_data(Vector<usize>{}, { 1.3f });
	h.set(i);

	REQUIRE(h.at({}) == 1.3f);
}

TEST_CASE("Tensor set() does not attach tensors", "[tensor]")
{
	SECTION("Scalar");
	Tensor a{ Vector<usize>{} };
	auto b = Tensor::from_data(Vector<usize>{}, { 1.3f });
	a.set(b);
	REQUIRE(a.is_detached());
	REQUIRE(b.is_detached());
	a.at({}).set(2.4f);
	REQUIRE(a.at({}) == 2.4f);
	REQUIRE(b.at({}) == 1.3f);

	SECTION("Rank 3 tensor");
	Tensor c{ {2, 2, 2} };
	auto d = Tensor::from_data({ 2, 2, 2 }, { 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f });
	c.set(d);
	REQUIRE(c.is_detached());
	REQUIRE(c.is_detached());
	c.at({ {0, 1, 0} }).set(2.4f);
	REQUIRE(c.at({ { 0, 1, 0 } }) == 2.4f);
	REQUIRE(d.at({ { 0, 1, 0 } }) == 1.3f);
}

TEST_CASE("Tensor set() throws when setting empty non-detached tensor", "[tensor]")
{
	Tensor a{ Vector<usize>{} };
	auto b = a;
	auto c = Tensor::zeroes(Vector<usize>{});
	REQUIRE_THROWS(a.set(c));
	REQUIRE_THROWS(b.set(c));
}

TEST_CASE("Tensor set() throws for incompatible dims", "[tensor]")
{
	SECTION("Scalar with vector");
	Tensor a{ Vector<usize>{} };
	auto b = Tensor::zeroes(Vector<usize>{});
	auto c = Tensor::from_data({ 3 }, { 1.3f, 1.4f, 1.5f });
	REQUIRE_THROWS(a.set(c));
	REQUIRE_THROWS(b.set(c));
	REQUIRE_THROWS(c.set(b));

	SECTION("Rank 3 tensor with different rank 3 tensor");
	Tensor d{ {2, 2, 2} };
	auto e = Tensor::from_data({ 1, 2, 2 }, { 1.1f, 1.2f, 1.3f, 1.4f });
	REQUIRE_THROWS(d.set(e));
}

TEST_CASE("Tensor set() throws for empty tensor argument", "[tensor]")
{
	Tensor a{ Vector<usize>{} };
	auto b = Tensor::zeroes(Vector<usize>{});
	Tensor c{ {2, 2, 2} };
	auto d = Tensor::from_data({ 2, 2, 2 }, { 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f });

	Tensor e{ Vector<usize>{} };
	Tensor f{ {2, 2, 2} };

	REQUIRE_THROWS(a.set(e));
	REQUIRE_THROWS(a.set(f));

	REQUIRE_THROWS(b.set(e));
	REQUIRE_THROWS(b.set(f));

	REQUIRE_THROWS(c.set(e));
	REQUIRE_THROWS(c.set(f));

	REQUIRE_THROWS(d.set(e));
	REQUIRE_THROWS(d.set(f));
}

TEST_CASE("Tensor make_zero() and make_normal() give correct results including for attached tensors", "[tensor]")
{
	Tensor a{ {3, 2} };
	auto b = Tensor::normals({ 3, 2 }, 0, 1);
	auto c = b;
	auto d = Tensor::zeroes({ 3, 2 });
	auto e = d;

	auto zeroes = Tensor::zeroes({ 3, 2 });
	a.make_zero();
	c.make_zero();
	d.make_normal(1.2f, 3.4f);

	for (const auto &position : zeroes.positions())
	{
		REQUIRE(a.at(position) == zeroes.at(position));
		REQUIRE(b.at(position) == zeroes.at(position));
		REQUIRE(c.at(position) == zeroes.at(position));
		REQUIRE(d.at(position) != zeroes.at(position));
		REQUIRE(e.at(position) != zeroes.at(position));
	}
}

TEST_CASE("Tensor make_zero() and make_normal() do not affect previously attached tensors", "[tensor]")
{
	float value;

	auto a = Tensor::from_data({ 3, 2 }, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f });
	auto b = a;
	auto c = a;
	auto d = a.copy();
	b.detach();

	a.make_zero();
	value = 0.f;
	for (const auto &position : a.positions())
	{
		++value;
		REQUIRE(float_approx_eq(b.at(position), value));
		REQUIRE(float_approx_eq(d.at(position), value));
	}
	REQUIRE(value == 6.f);

	c.make_normal(0, 1);
	value = 0.f;
	for (const auto &position : a.positions())
	{
		++value;
		REQUIRE(float_approx_eq(b.at(position), value));
		REQUIRE(float_approx_eq(d.at(position), value));
	}
	REQUIRE(value == 6.f);
}

TEST_CASE("Tensor apply_inplace() applies correctly", "[tensor]")
{
	auto a = Tensor::zeroes({ 2, 3 });
	auto b = a;
	auto c = Tensor::from_data(Vector<usize>{}, { 1.1f });
	auto d = Tensor::from_data({ 2, 2 }, { 1, 2, 3, 4 });
	auto e = d;

	a.apply_inplace([](float input) { return input + 1.f; });
	b.apply_inplace([](float input) { return input + 1.f; });

	c.apply_inplace([](float input) { return 0.f; });
	d.apply_inplace([](float input) { return 0.f; });

	REQUIRE(std::all_of(
		a.positions().begin(), a.positions().end(),
		[a](const auto &position) { return float_approx_eq(a.at(position), 2.f); }
	));

	REQUIRE(c.at({}) == 0.f);

	REQUIRE(std::all_of(
		d.positions().begin(), d.positions().end(),
		[d](const auto &position) { return d.at(position) == 0.f; }
	));

	REQUIRE(std::all_of(
		e.positions().begin(), e.positions().end(),
		[e](const auto &position) { return e.at(position) == 0.f; }
	));
}

TEST_CASE("Tensor apply_inplace() throws on empty tensor", "[tensor]")
{
	Tensor a{ Vector<usize>{} };
	Tensor b{ {2, 2, 3} };

	REQUIRE_THROWS(a.apply_inplace([](float input) { return input + 1.f; }));
	REQUIRE_THROWS(b.apply_inplace([](float input) { return input + 1.f; }));

	REQUIRE_THROWS(a.apply_inplace([](float input) { return 0.f; }));
	REQUIRE_THROWS(b.apply_inplace([](float input) { return 0.f; }));
}
