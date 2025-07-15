#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iomanip>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

using namespace std;

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
    double parameters;

    double get_dtype_divider() const {
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

ModelConfig parseConfig(const json& j, double p) {
    ModelConfig mc;

    if (j.contains("text_config")) {
        // this shouldnt be here but just in case
        return parseConfig(j["text_config"], p);
    }

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


double inputBuffer(int context, const ModelConfig& mc, int bsz) {
    int inp_tokens = bsz;
    int inp_embd = mc.hidden_size * bsz;
    int inp_pos = bsz;
    int inp_KQ_mask = context * bsz;
    int inp_K_shift = context;
    int inp_sum = bsz;

    return inp_tokens + inp_embd + inp_pos + inp_KQ_mask + inp_K_shift + inp_sum;
}


double computeBuffer(int context, const ModelConfig& mc, int bsz) {
    if (bsz != 512) {
        cerr << "Warning: batch size other than 512 is currently not supported for the compute buffer calculation" << endl;
        bsz = 512; // forcibly set
    }
    return (context / 1024.0 * 2.0 + 0.75) * mc.num_attention_heads * 1024 * 1024;
}



double kvCache(int context, const ModelConfig& mc, int cache_bit) {
    double n_gqa = (double)mc.num_attention_heads / mc.num_key_value_heads;
    double n_embd_gqa = mc.hidden_size / n_gqa;
    double n_elements = n_embd_gqa * (mc.num_hidden_layers * context);
    double size = 2.0 * n_elements;
    return size * (cache_bit / 8.0);
}


double contextSize(int context, const ModelConfig& mc, int bsz, int cache_bit) {
    return inputBuffer(context, mc, bsz) + kvCache(context, mc, cache_bit) + computeBuffer(context, mc, bsz);
}


double modelSize(const ModelConfig& mc, double bpw) {
    return (mc.parameters * bpw) / 8.0;
}


int main(int argc, char* argv[]) {

    /*
    cli mode input format

	argv[0] = executable name
	argv[1] = path to config.json
	argv[2] = parameters in billions
	argv[3] = quant format (gguf or exl2)
	argv[4] = ctx
	argv[5] = kv cache bit size (if exl2)
	argv[5] = batch size (if gguf) (exclusive)
	argv[6] = bpw (if exl2)
	argv[6] = quant size (if gguf) (exclusive)
    */

    // these get actually set later
    string configPath{};
    double p{};
    string quantFormat{};
    int context = 8192;
    int bsz = 512;
    int cache_bit = 16;
    double bpw = 0;
    string quantSize{};

    // gui mode onramp
    if (argc != 7) {
        cout << "If you were looking for the CLI mode, please use the format below." << endl;
        cout << "Usage: " << argv[0] << " <path_to_config_json>" << " <parameters (float, billions)>" << " <quant_format (gguf or exl2)>" << " <context_size (int)>"
            << " [<kv_cache_bit_size (if exl2, 16/8/4)>" << " <batch_size (if gguf, int)>]" << " [<bpw (if exl2, float)>" << " <quant_size (if gguf, string)>]" <<
            "\nwhere you only include one from each square bracket pair depending on your desired quant format." << '\n' << endl;
        
        cout << "Enter your model config path (local):\n";
        cin >> configPath;
        replace(configPath.begin(), configPath.end(), '\\', '/');
        configPath.erase(std::remove(configPath.begin(), configPath.end(), '\"'), configPath.end());

        cout << "Enter number of parameters (in billions):\n";
        cin >> p;
        p *= 1000000000;

		cout << "Enter quant format (gguf or exl2):\n";
        cin >> quantFormat;
        std::transform(quantFormat.begin(), quantFormat.end(), quantFormat.begin(), ::tolower);

        cout << "Enter context size (default 8192):\n";
        // handle newline behaviour
        string input;
        cin.ignore();
        getline(cin, input);
        if (!input.empty()) {
            try {
                context = stoi(input);
            }
            catch (...) {
                cout << "Invalid input for context size, using default 8192.\n";
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

            // matching is case sensitive in map!!
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
    }
    else {
        configPath = argv[1];
        
        p = atof(argv[2]) * 1e9L;

        quantFormat = argv[3];

        context = atoi(argv[4]);

        if (quantFormat == "gguf") {
            bsz = atoi(argv[5]);
            quantSize = argv[6];
            bpw = gguf_quants.at(quantSize);
        } else if (quantFormat == "exl2") {
            cache_bit = atoi(argv[5]);
            bpw = atof(argv[6]);
        }
        else {
            cerr << "Unsupported quant format (" << quantFormat << "). Exiting." << endl;
            return 1;
		}
    }

    // read config file (this happens no matter if cli or not)
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

    // showtime
    try {
        double model_size = modelSize(mc, bpw);
        double context_size = contextSize(context, mc, bsz, cache_bit);
        double total_size = model_size + context_size;

        if (argc != 7) {
            cout << fixed << setprecision(3);
            cout << "\nResults (in GB):" << endl;
            cout << "  Model Size:   " << model_size / (1024 * 1024 * 1024) << " GB" << endl;
            cout << "  Context Size: " << context_size / (1024 * 1024 * 1024) << " GB" << endl;
            cout << "  Total Size:   " << total_size / (1024 * 1024 * 1024) << " GB" << endl;
        }
        else {
            cout << fixed << setprecision(8);
            std::cout << "{\n";
            std::cout << "  \"model_size\": " << model_size / (1024 * 1024 * 1024) << ",\n";
            std::cout << "  \"context_size\": " << context_size / (1024 * 1024 * 1024) << ",\n";
            std::cout << "  \"total_size\": " << total_size / (1024 * 1024 * 1024) << "\n";
            std::cout << "}" << std::endl;

        }
    }
    catch (exception& e) {
        cerr << "Error during calculation: " << e.what() << endl;
        return 1;
    }

    return 0;
}