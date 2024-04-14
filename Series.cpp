#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <torch/torch.h>

// Structure to store color information from CSV
struct ColorInfo {
    std::string color_name;
    int temperature_f;
    int temperature_c;
    std::string intuitive_interp;
    int dbz;
    std::string precip_rate_desc;
    std::string hex_code;
};

// Function to parse CSV file and store color information
std::vector<ColorInfo> parseCSV(const std::string& filename) {
    std::vector<ColorInfo> color_info;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open CSV file: " << filename << std::endl;
        return color_info;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        ColorInfo info;
        if (!(iss >> info.color_name >> info.temperature_f >> info.temperature_c 
              >> info.intuitive_interp >> info.dbz >> info.precip_rate_desc >> info.hex_code)) {
            std::cerr << "Error: Failed to parse line in CSV file: " << line << std::endl;
            continue;
        }
        color_info.push_back(info);
    }

    file.close();
    return color_info;
}

// Function to read tensor file and return pixel values
std::vector<int> readTensorFile(const std::string& filename) {
    std::vector<int> pixel_values;

    // Load tensor from file
    torch::Tensor tensor = torch::load(filename);

    // Ensure tensor is in uint8 format
    if (tensor.scalar_type() != torch::kUInt8) {
        std::cerr << "Error: Tensor data type is not uint8" << std::endl;
        return pixel_values;
    }

    // Flatten the tensor and copy data to pixel_values vector
    tensor = tensor.view({-1});
    pixel_values.resize(tensor.size(0));
    std::memcpy(pixel_values.data(), tensor.data_ptr<uint8_t>(), tensor.numel());

    return pixel_values;
}

// Function to match pixel values with temperature scale and calculate average temperature
void calculateAverageTemperature(const std::string& tensor_filename, const std::vector<ColorInfo>& color_info, int num_pixels) {
    // Read pixel values from tensor file
    std::vector<int> pixel_values = readTensorFile(tensor_filename);

    // Perform mapping and calculate average temperature
    int sum_temperature_f = 0;
    int sum_temperature_c = 0;
    int count = 0;
    for (int pixel_value : pixel_values) {
        // Map pixel value to color information
        for (const auto& info : color_info) {
            if (info.hex_code == pixel_value) {
                sum_temperature_f += info.temperature_f;
                sum_temperature_c += info.temperature_c;
                count++;
                break;
            }
        }
        // Break if required number of pixels reached
        if (count >= num_pixels)
            break;
    }

    // Calculate average temperature
    double average_temperature_f = static_cast<double>(sum_temperature_f) / count;
    double average_temperature_c = static_cast<double>(sum_temperature_c) / count;

    std::cout << "Average temperature of " << num_pixels << " pixels (Fahrenheit): " << average_temperature_f << std::endl;
    std::cout << "Average temperature of " << num_pixels << " pixels (Celsius): " << average_temperature_c << std::endl;
}

int main() {
    // Read CSV file
    std::vector<ColorInfo> color_info = parseCSV("weather.csv");

    // Specify tensor file and number of pixels to process
    std::string tensor_filename = "image_patch_tensor.pt";
    int num_pixels = 1000 * 1000; // Assuming a square of 1000x1000 pixels

    // Calculate average temperature
    auto start_time = std::chrono::steady_clock::now();
    calculateAverageTemperature(tensor_filename, color_info, num_pixels);
    auto end_time = std::chrono::steady_clock::now();

    // Calculate elapsed time
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    std::cout << "Elapsed time: " << elapsed_seconds << " seconds" << std::endl;

    return 0;
}
