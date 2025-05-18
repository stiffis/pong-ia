#include "../include/utec/algebra/Tensor.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>

void test1() {
    utec::algebra::Tensor<int, 2> t(2, 3);
    t.fill(7);
    int x = t(1, 2);
    bool res = (x == 7);
    std::cout << "test1: " << (res ? "Passed" : "Failed") << std::endl;
}

void test2() {
    utec::algebra::Tensor<int, 2> t2(2, 3);
    std::array<size_t, 2> xd = {3, 2};
    t2.reshape(3, 2);
    int y = t2[5];
    bool res = (y == t2(2, 1));
    std::cout << "test2: " << (res ? "Passed" : "Failed") << std::endl;
}

void test3() { // BUG: this shit need to throw an exception(error).
    utec::algebra::Tensor<int, 3> t3(2, 2, 2);
    t3.reshape(2, 4, 1);
    bool res = (t3.shape() == std::array<size_t, 3>{2, 4, 1});
    std::cout << "test3: " << (res ? "Passed" : "Failed") << std::endl;
}

void test4() {
    utec::algebra::Tensor<double, 2> a(2, 2), b(2, 2);
    a(0, 1) = 5.5;
    b.fill(2.0);
    auto sum = a + b;
    auto diff = sum - b;
    bool res = (sum(0, 1) == 7.5 && diff(0, 1) == 5.5);
    std::cout << "test4: " << (res ? "Passed" : "Failed") << std::endl;
}

void test5() {
    utec::algebra::Tensor<float, 1> v(3);
    v.fill(2.0f);
    auto scaled = v * 4.0f;
    utec::algebra::Tensor<int, 3> cube(2, 2, 2);
    cube.fill(1);
    auto cube2 = cube * cube;
    bool res = (scaled(2) == 8.0f && cube2(1, 1, 1) == 1);
    std::cout << "test5: " << (res ? "Passed" : "Failed") << std::endl;
}

void test6() {
    utec::algebra::Tensor<int, 2> m(2, 1);
    m(0, 0) = 3;
    m(1, 0) = 4;
    utec::algebra::Tensor<int, 2> n(2, 3);
    n.fill(5);
    auto p = m * n;
    bool res = (p(0, 2) == 15 && p(1, 1) == 20);
    std::cout << "test6: " << (res ? "Passed" : "Failed") << std::endl;
}

void test7() {
    utec::algebra::Tensor<int, 2> m2(2, 3);
    auto mt = m2.transpose_2d();
    bool res =
        (mt.shape() == std::array<size_t, 2>{3, 2} && mt(0, 1) == mt(1, 0));
    std::cout << "test7: " << (res ? "Passed" : "Failed") << std::endl;
}

int main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    return 0;
}
