#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

using namespace std;

// Quant sizes & values from your gguf_quants map
const map<string, double> gguf_quants{
    {"IQ1_S", 1.56},
    {"IQ2_XXS", 2.06},
    {"IQ2_XS", 2.31},
    {"IQ2_S", 2.5},
    {"IQ2_M", 2.7},
    {"IQ3_XXS", 3.06},
    {"IQ3_XS", 3.3},
    {"Q2_K", 3.35},
    {"Q3_K_S", 3.5},
    {"IQ3_S", 3.5},
    {"IQ3_M", 3.7},
    {"Q3_K_M", 3.91},
    {"Q3_K_L", 4.27},
    {"IQ4_XS", 4.25},
    {"IQ4_NL", 4.5},
    {"Q4_0", 4.55},
    {"Q4_K_S", 4.58},
    {"Q4_K_M", 4.85},
    {"Q5_0", 5.54},
    {"Q5_K_S", 5.54},
    {"Q5_K_M", 5.69},
    {"Q6_K", 6.59},
    {"Q8_0", 8.5}
};

struct ModelConfig {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    std::string torch_dtype;
    float parameters;

    double get_dtype_divider() const {
        // Extract digits from torch_dtype's string and divide by 8
        // e.g. "torch.float16" -> 16/8 = 2
        string digits_only;
        for (char c : torch_dtype) {
            if (isdigit(c)) {
                digits_only.push_back(c);
            }
        }
        if (digits_only.empty())
            throw runtime_error("Could not parse torch_dtype number");
        double val = stod(digits_only) / 8.0;
        return val;
    }
};

ModelConfig parseConfig(const json& j, float p) {
    ModelConfig mc;

    if (j.contains("text_config")) {
        // If text_config key exists, use that instead
        return parseConfig(j["text_config"], p);
    }

    cerr << "hidden_size present: " << j.contains("hidden_size") << endl;
    cerr << "num_attention_heads present: " << j.contains("num_attention_heads") << endl;
    cerr << "num_key_value_heads present: " << j.contains("num_key_value_heads") << endl;
    cerr << "num_hidden_layers present: " << j.contains("num_hidden_layers") << endl;
    cerr << "torch_dtype present: " << j.contains("torch_dtype") << endl;

    if (!j.contains("hidden_size") || !j.contains("num_attention_heads") || !j.contains("num_key_value_heads")
        || !j.contains("num_hidden_layers") || !j.contains("torch_dtype")) {
        throw runtime_error("Some required keys are missing in the config.json");
    }

    mc.hidden_size = j["hidden_size"].get<int>();
    mc.num_attention_heads = j["num_attention_heads"].get<int>();
    mc.num_key_value_heads = j["num_key_value_heads"].get<int>();
    mc.num_hidden_layers = j["num_hidden_layers"].get<int>();
    mc.torch_dtype = j["torch_dtype"].get<string>();
    mc.parameters = p;

    return mc;
}

/*
inputBuffer calculation copied from JS:
  const inp_tokens = bsz
  const inp_embd = model_config["hidden_size"] * bsz
  const inp_pos = bsz
  const inp_KQ_mask = context * bsz
  const inp_K_shift = context
  const inp_sum = bsz

  return inp_tokens + inp_embd + inp_pos + inp_KQ_mask + inp_K_shift + inp_sum
*/
double inputBuffer(int context, const ModelConfig& mc, int bsz) {
    int inp_tokens = bsz;
    int inp_embd = mc.hidden_size * bsz;
    int inp_pos = bsz;
    int inp_KQ_mask = context * bsz;
    int inp_K_shift = context;
    int inp_sum = bsz;

    return inp_tokens + inp_embd + inp_pos + inp_KQ_mask + inp_K_shift + inp_sum;
}

/*
computeBuffer calculation ported from JS:
  (context / 1024 * 2 + 0.75) * model_config["num_attention_heads"] * 1024 * 1024
*/
double computeBuffer(int context, const ModelConfig& mc, int bsz) {
    if (bsz != 512) {
        cerr << "Warning: batch size other than 512 is currently not supported for the compute buffer calculation" << endl;
        bsz = 512; // forcibly set
    }
    return (context / 1024.0 * 2.0 + 0.75) * mc.num_attention_heads * 1024 * 1024;
}


/*
kvCache calculation from JS:
  n_gqa = num_attention_heads / num_key_value_heads
  n_embd_gqa = hidden_size / n_gqa
  n_elements = n_embd_gqa * (num_hidden_layers * context)
  size = 2 * n_elements
  return size * (cache_bit / 8)
*/
double kvCache(int context, const ModelConfig& mc, int cache_bit) {
    double n_gqa = (double)mc.num_attention_heads / mc.num_key_value_heads;
    double n_embd_gqa = mc.hidden_size / n_gqa;
    double n_elements = n_embd_gqa * (mc.num_hidden_layers * context);
    double size = 2.0 * n_elements;
    return size * (cache_bit / 8.0);
}

/*
contextSize from JS:
inputBuffer(context, mc, bsz) + kvCache(context, mc, cache_bit) + computeBuffer(context, mc, bsz)
*/
double contextSize(int context, const ModelConfig& mc, int bsz, int cache_bit) {
    return inputBuffer(context, mc, bsz) + kvCache(context, mc, cache_bit) + computeBuffer(context, mc, bsz);
}

/*
modelSize calculation from JS:
model_config["parameters"] * bpw / 8
*/
double modelSize(const ModelConfig& mc, double bpw) {
    return (mc.parameters * bpw) / 8.0;
}


int main(int argc, char* argv[]) {

    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <path_to_config_json>" << " <parameters (float, billions)>"  << endl;
        return 1;
    }

    string configPath = argv[1];
    float p = atof(argv[2]) * 1000000000;

    // Read config file
    ifstream file(configPath);
    if (!file.is_open()) {
        cerr << "Failed to open config file: " << configPath << endl;
        return 1;
    }

    json configJson;
    try {
        file >> configJson;
    }
    catch (exception& e) {
        cerr << "Failed to parse JSON: " << e.what() << endl;
        return 1;
    }

    ModelConfig mc;
    try {
        mc = parseConfig(configJson, p);
    }
    catch (exception& e) {
        cerr << "Error parsing model config: " << e.what() << endl;
        return 1;
    }

    // Determine dtype divider and adjust parameters accordingly if parameters is 0 or not matching
    // (in JS code, they calculate parameters from file metadata / dtype divider. We expect parameters to be given directly)
    // Just to keep compatibility, we do nothing here, assuming parameters are correct.

    // Ask the user for inputs

    cout << "Enter quant format (gguf or exl2): ";
    string quantFormat{};
    cin >> quantFormat;
    std::transform(quantFormat.begin(), quantFormat.end(), quantFormat.begin(), ::tolower);

    int context = 8192;
    int bsz = 512;
    int cache_bit = 16;
    double bpw = 0;

    cout << "Enter context size (default 8192): ";
    string input;
    cin.ignore(); // flush newline from input buffer
    getline(cin, input);
    if (!input.empty()) {
        try {
            context = stoi(input);
        }
        catch (...) {
            cout << "Invalid input for context size, using default 8192." << endl;
            context = 8192;
        }
    }

    if (quantFormat == "gguf") {
        cout << "Enter quantization size (default Q4_K_S). Valid options:\n";
        for (auto& kv : gguf_quants) {
            cout << " - " << kv.first << "\n";
        }
        cout << "Quantization size: ";
        string quantSize;
        getline(cin, quantSize);
        if (quantSize.empty()) quantSize = "Q4_K_S";
        else quantSize.erase(remove_if(quantSize.begin(), quantSize.end(), ::isspace), quantSize.end()); // trim spaces

        // Make all uppercase for safety & matching keys - but keep format intact
        // Matching is case sensitive in map, so we do exact match
        if (gguf_quants.find(quantSize) == gguf_quants.end()) {
            cout << "Invalid quantization size entered, defaulting to Q4_K_S" << endl;
            quantSize = "Q4_K_S";
        }

        cout << "Enter batch size (default 512): ";
        string batchStr;
        getline(cin, batchStr);
        if (!batchStr.empty()) {
            try {
                int tmpbsz = stoi(batchStr);
                if (tmpbsz > 0) bsz = tmpbsz;
            }
            catch (...) {
                cout << "Invalid input for batch size, using default 512." << endl;
            }
        }

        bpw = gguf_quants.at(quantSize);

    }
    else if (quantFormat == "exl2") {
        cout << "Enter BPW (bits per weight) (default 4.5): ";
        string bpwStr;
        getline(cin, bpwStr);
        if (!bpwStr.empty()) {
            try {
                bpw = stod(bpwStr);
                if (bpw <= 0) {
                    cout << "Invalid BPW; must be positive. Using default 4.5." << endl;
                    bpw = 4.5;
                }
            }
            catch (...) {
                cout << "Invalid BPW input; using default 4.5." << endl;
                bpw = 4.5;
            }
        }
        else {
            bpw = 4.5;
        }

        cout << "Enter KV Cache bit size (16, 8, or 4) (default 16): ";
        string kvStr;
        getline(cin, kvStr);
        if (!kvStr.empty()) {
            try {
                int v = stoi(kvStr);
                if (v == 16 || v == 8 || v == 4) {
                    cache_bit = v;
                }
                else {
                    cout << "Invalid KV cache bit size, defaulting to 16." << endl;
                    cache_bit = 16;
                }
            }
            catch (...) {
                cout << "Invalid KV cache bit size, defaulting to 16." << endl;
                cache_bit = 16;
            }
        }
        else {
            cache_bit = 16;
        }

    }
    else {
        cout << "Unsupported quant format (" << quantFormat << "). Exiting." << endl;
        return 1;
    }

    try {
        double model_size = modelSize(mc, bpw);
        double context_size = contextSize(context, mc, bsz, cache_bit);
        double total_size = model_size + context_size;

        cout << "\nResults (in GB):" << endl;
        cout << "  Model Size:   " << model_size / (1024 * 1024 * 1024) << " GB" << endl;
        cout << "  Context Size: " << context_size / (1024 * 1024 * 1024) << " GB" << endl;
        cout << "  Total Size:   " << total_size / (1024 * 1024 * 1024) << " GB" << endl;
    }
    catch (exception& e) {
        cerr << "Error during calculation: " << e.what() << endl;
        return 1;
    }

    return 0;
}