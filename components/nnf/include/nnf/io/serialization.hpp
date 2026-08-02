#pragma once

#include <bit>
#include <fstream>
#include <limits>
#include <nnf/utils/base_types.hpp>
#include <nnf/utils/vector.hpp>

namespace nnf::io
{
    void require_compatible_floats()
    {
	    // Ensure compatibility between system used to read the files and system used to write the files
	    // These lines can be removed if you are only reading and writing on the same system
	    static_assert(std::numeric_limits<float>::is_iec559, "Requires IEEE 754 floats");
	    static_assert(sizeof(float) == 4, "Requires 32-bit floats");
	    static_assert(std::endian::native == std::endian::little, "Requires little-endianness");
    }
    
    void read_floats(Vector<float> &floats, usize count, std::ifstream &stream)
    {
        require_compatible_floats();
        floats.resize(count);
        stream.read(
            reinterpret_cast<char *>(floats.data()),
            static_cast<std::streamsize>(floats.size() * sizeof(float))
        );
    }
    
    void write_floats(VectorView<float> floats, std::ofstream &stream)
    {
        require_compatible_floats();
        stream.write(
            reinterpret_cast<const char *>(floats.data()),
            static_cast<std::streamsize>(floats.size_bytes())
        );
    }
}
