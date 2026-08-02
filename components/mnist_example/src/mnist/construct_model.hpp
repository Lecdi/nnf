#pragma once

#include <memory>
#include <nnf/ml/models.hpp>

namespace mnist
{
    std::unique_ptr<nnf::ml::Sequential> construct_model();
}
