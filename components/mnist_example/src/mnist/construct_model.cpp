#include <mnist/construct_model.hpp>

#include <memory>
#include <utility>
#include <nnf/io/logging.hpp>
#include <nnf/ml/layers.hpp>
#include <nnf/ml/models.hpp>
#include <nnf/ml/losses.hpp>
#include <nnf/utils/vector.hpp>

namespace mnist
{
    std::unique_ptr<nnf::ml::Sequential> construct_model()
    {
        nnf::Vector<std::unique_ptr<nnf::ml::Layer>> layers{};
        layers.push_back(nnf::ml::dense_relu_layer(28 * 28, 128));
        layers.push_back(nnf::ml::dense_relu_layer(128, 64));
        layers.push_back(nnf::ml::dense_no_activation_layer(64, 10));

        return std::make_unique<nnf::ml::Sequential>(
            28 * 28,
            10,
            std::move(layers),
            nnf::ml::softmax_crossentropy_loss(),
            nnf::io::stdout_model_logger()
        );
    }
}
