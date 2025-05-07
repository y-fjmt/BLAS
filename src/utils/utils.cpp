#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <regex>
#include <cstdlib>
#include <chrono>

using namespace std;

double _mm_gflops(double sec, int N) {
    return (2.0*N*N*N) / (sec * 1e9);
}


string cpu_name() {
    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    string model_name;
    regex model_regex("model name\\s*:\\s*(.*)");
    regex physid_regex("physical id\\s*:\\s*(\\d+)");
    set<int> socket_ids;

    if (cpuinfo.is_open()) {
        while (getline(cpuinfo, line)) {
            smatch match;
            if (regex_search(line, match, physid_regex)) {
                socket_ids.insert(stoi(match[1].str()));
            }
            if (regex_search(line, match, model_regex)) {
                model_name = match[1].str();
            }
        }
        cpuinfo.close();
    } else {
        cerr << "Unable to open /proc/cpuinfo" << endl;
        return "Unknown";
    }
    string amount = " * " + to_string(socket_ids.size());
    return (model_name.empty())? "Unknown":model_name+amount;
}
