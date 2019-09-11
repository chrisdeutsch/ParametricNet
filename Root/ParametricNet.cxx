#include "ParametricNet/ParametricNet.h"

#include <fstream>

#include "lwtnn/LightweightGraph.hh"
#include "lwtnn/parse_json.hh"


ParametricNet::ParametricNet() :
  m_input_layer_name("input_layer"),
  m_output_layer_name("output_layer"),
  m_output_node_name("sig_prob"),
  m_parameter_name("mass") {
}

ParametricNet::~ParametricNet() {
}

void ParametricNet::init(const std::string &filename_even,
                         const std::string &filename_odd) {
  std::ifstream input_file(filename_even);
  lwt::GraphConfig config_even = lwt::parse_json_graph(input_file);
  input_file.close();

  input_file.open(filename_odd);
  lwt::GraphConfig config_odd = lwt::parse_json_graph(input_file);
  input_file.close();

  m_graph_even = std::make_unique<lwt::LightweightGraph>(config_even, m_output_layer_name);
  m_graph_odd = std::make_unique<lwt::LightweightGraph>(config_odd, m_output_layer_name);

  // Create empty map for input layer
  m_val_map = &m_node_map[m_input_layer_name];
}

void ParametricNet::set_vars(double DRTauTau, double dRBB, double mMMC,
                             double mBB, double mHH) {
  (*m_val_map)["DRTauTau"] = DRTauTau;
  (*m_val_map)["dRBB"] = dRBB;
  (*m_val_map)["mMMC"] = mMMC;
  (*m_val_map)["mBB"] = mBB;
  (*m_val_map)["mHH"] = mHH;
}

float ParametricNet::evaluate(float parameter, Fold fold) {
  (*m_val_map)[m_parameter_name] = parameter;

  const lwt::LightweightGraph *graph = nullptr;
  if (fold == Fold::Even) {
    graph = m_graph_odd.get();
  } else if (fold == Fold::Odd) {
    graph = m_graph_even.get();
  } else {
    return -999.0;
  }

  const auto &result = graph->compute(m_node_map);
  return result.at(m_output_node_name);
}
