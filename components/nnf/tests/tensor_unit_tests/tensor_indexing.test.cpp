#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <nnf/tensor/indexing.hpp>
#include <nnf/tensor/tensor.hpp>
#include <nnf/utils/vector.hpp>

using namespace nnf;

TEST_CASE("Tensor positions() range iterates through all possible positions in order")
{
	usize current_index;

	SECTION("Scalar");
	Tensor a{ Vector<usize>{} };
	current_index = 0;
	for (const auto &position : a.positions())
	{
		REQUIRE(current_index == 0);
		REQUIRE(position.size() == 0);
		++current_index;
	}

	SECTION("Rank 3 tensor");
	Tensor b{ {1, 5, 8} };
	current_index = 0;
	Vector<usize> expected_0{ 0, 0, 0 };
	Vector<usize> expected_8{ 0, 1, 0 };
	Vector<usize> expected_33{ 0, 4, 1 };
	Vector<usize> actual_0, actual_8, actual_33;
	for (const auto &position : b.positions())
	{
		if (current_index == 0)
			actual_0 = { position.begin(), position.end() };
		if (current_index == 8)
			actual_8 = { position.begin(), position.end() };
		if (current_index == 33)
			actual_33 = { position.begin(), position.end() };
		++current_index;
	}
	REQUIRE(current_index == 40);
	REQUIRE(actual_0 == expected_0);
	REQUIRE(actual_8 == expected_8);
	REQUIRE(actual_33 == expected_33);
}

TEST_CASE("Tensor zeroes() factory produces all zeroes", "[tensor]")
{
	SECTION("Scalar");
	auto a = Tensor::zeroes(Vector<usize>{});
	REQUIRE(a.at({}) == 0);

	SECTION("Rank 3 tensor");
	auto b = Tensor::zeroes({ 1, 5, 8 });
	REQUIRE(std::all_of(b.positions().begin(), b.positions().end(),
		[b](const auto &position) { return b.at(position) == 0; }));
}

TEST_CASE("Tensor normals() factory produces no zeroes", "[tensor]")
{
	SECTION("Matrix");
	auto a = Tensor::normals({ 3, 4 }, 0, 1);
	REQUIRE(std::all_of(
		a.positions().begin(), a.positions().end(),
		[a](const auto &position) {return a.at(position) != 0; }
	));
}

TEST_CASE("Tensor from_data() factory produces tensor with given values", "[tensor]")
{
	SECTION("Rank 3 tensor");
	auto a = Tensor::from_data({ 1, 2, 3 }, { 1, 2, 3, 4, 5, 6 });
	REQUIRE(a.at({ { 0, 0, 0 } }) == 1.f);
	REQUIRE(a.at({ { 0, 0, 1 } }) == 2.f);
	REQUIRE(a.at({ { 0, 0, 2 } }) == 3.f);
	REQUIRE(a.at({ { 0, 1, 0 } }) == 4.f);
	REQUIRE(a.at({ { 0, 1, 1 } }) == 5.f);
	REQUIRE(a.at({ { 0, 1, 2 } }) == 6.f);
}

TEST_CASE("Tensor slicing returns correct dimensions and values", "[tensor]")
{
	auto a = Tensor::from_data(
		{ 6, 3, 4 },
		{
			  0.f,   1.f,   2.f,   3.f,  10.f,  11.f,  12.f,  13.f,  20.f,  21.f,  22.f,  23.f,
			100.f, 101.f, 102.f, 103.f, 110.f, 111.f, 112.f, 113.f, 120.f, 121.f, 122.f, 123.f,
			200.f, 201.f, 202.f, 203.f, 210.f, 211.f, 212.f, 213.f, 220.f, 221.f, 222.f, 223.f,
			300.f, 301.f, 302.f, 303.f, 310.f, 311.f, 312.f, 313.f, 320.f, 321.f, 322.f, 323.f,
			400.f, 401.f, 402.f, 403.f, 410.f, 411.f, 412.f, 413.f, 420.f, 421.f, 422.f, 423.f,
			500.f, 501.f, 502.f, 503.f, 510.f, 511.f, 512.f, 513.f, 520.f, 521.f, 522.f, 523.f
		}
	);

	SECTION("Slice with ellipsis only");
	auto b = a({ ellipsis });
	REQUIRE(std::equal(a.view_dims().begin(), a.view_dims().end(), b.view_dims().begin(), b.view_dims().end()));
	REQUIRE(std::all_of(
		a.positions().begin(), a.positions().end(),
		[a, b](const auto &position) { return a.at(position) == b.at(position); }
	));

	SECTION("Slice with three-argument index and ellipsis");
	auto c = a({ {{0, 3, 2}}, ellipsis });
	auto c_dims = c.view_dims();
	REQUIRE(c_dims.size() == 3);
	REQUIRE(c_dims[0] == 2);
	REQUIRE(c_dims[1] == 3);
	REQUIRE(c_dims[2] == 4);
	REQUIRE(c.at({ { 1, 0, 2 } }) == 202.f);

	SECTION("Slice with two-argument index and ellipsis");
	auto d = a.reshaped({ 1, 1, 1, 6, 3, 4 })({ ellipsis, {{1, 5}}, {{0, 0}}, {{0, 0}} });
	auto d_dims = d.view_dims();
	REQUIRE(d_dims.size() == 6);
	REQUIRE(d_dims[0] == 1);
	REQUIRE(d_dims[1] == 1);
	REQUIRE(d_dims[2] == 1);
	REQUIRE(d_dims[3] == 4);
	REQUIRE(d_dims[4] == 3);
	REQUIRE(d_dims[5] == 4);
	REQUIRE(d.at({ { 0, 0, 0, 3, 1, 2 } }) == 412.f);

	SECTION("Slice with one-argument indices");
	auto e = a({ {0}, {1}, {3} });
	auto e_dims = e.view_dims();
	REQUIRE(e_dims.size() == 3);
	REQUIRE(e_dims[0] == 1);
	REQUIRE(e_dims[1] == 1);
	REQUIRE(e_dims[2] == 1);
	REQUIRE(e.at({ { 0, 0, 0 } }) == 13.f);
}

TEST_CASE("Modifying tensor slice modifies original tensor", "[tensor]")
{
	auto a = Tensor::from_data(
		{ 6, 3, 4 },
		{
			  0.f,   1.f,   2.f,   3.f,  10.f,  11.f,  12.f,  13.f,  20.f,  21.f,  22.f,  23.f,
			100.f, 101.f, 102.f, 103.f, 110.f, 111.f, 112.f, 113.f, 120.f, 121.f, 122.f, 123.f,
			200.f, 201.f, 202.f, 203.f, 210.f, 211.f, 212.f, 213.f, 220.f, 221.f, 222.f, 223.f,
			300.f, 301.f, 302.f, 303.f, 310.f, 311.f, 312.f, 313.f, 320.f, 321.f, 322.f, 323.f,
			400.f, 401.f, 402.f, 403.f, 410.f, 411.f, 412.f, 413.f, 420.f, 421.f, 422.f, 423.f,
			500.f, 501.f, 502.f, 503.f, 510.f, 511.f, 512.f, 513.f, 520.f, 521.f, 522.f, 523.f
		}
	);

	auto b = Tensor::zeroes({ 1, 1, 1, 4, 3, 4 });

	a.reshaped({ 1, 1, 1, 6, 3, 4 })({ ellipsis, {{1, 5}}, {{0, 0}}, {{0, 0}} }).set(b);
	REQUIRE(a.at({ {3, 2, 1} }) == 0.f);
	REQUIRE(a.at({ {1, 2, 1} }) == 0.f);
	REQUIRE(a.at({ {1, 0, 0} }) == 0.f);
	REQUIRE(a.at({ {4, 0, 3} }) == 0.f);
	REQUIRE(a.at({ {0, 0, 3} }) == 3.f);
	REQUIRE(a.at({ {5, 0, 3} }) == 503.f);

	auto c = Tensor::zeroes({ 3, 2 });
	c({ {0}, {1}, ellipsis }).set(Tensor::from_data({ 1, 1 }, { 5.f }));
	REQUIRE(c.at({ {0, 1} }) == 5.f);
}

TEST_CASE("Tensor slicing throws for invalid slice", "[tensor]")
{
	REQUIRE_THROWS(Tensor({ 3, 2 })({ {1}, {0}, {0}, ellipsis }));
	REQUIRE_THROWS(Tensor({ 3, 2 })({ {1} }));
	REQUIRE_THROWS(Tensor({ 3, 2 })({ {3}, {0} }));
	REQUIRE_THROWS(Tensor({ 3, 2 })({ {3}, {0}, ellipsis }));
}
