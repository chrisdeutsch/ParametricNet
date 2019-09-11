#ifndef PARAMETRICNET_H_
#define PARAMETRICNET_H_

#include <string>
#include <map>
#include <memory>

// Forward declaration
namespace lwt {
  class LightweightGraph;
}

class ParametricNet {

public:
  enum class Fold {Even, Odd};

  ParametricNet();
  ~ParametricNet();

  void init(const std::string &filename_even,
            const std::string &filename_odd);

  void set_vars(double DRTauTau, double dRBB, double mMMC,
                double mBB, double mHH);

  float evaluate(float parameter, Fold fold);

private:
  using ValueMap = std::map<std::string, double>;
  using NodeMap = std::map<std::string, ValueMap>;

  const std::string m_input_layer_name;
  const std::string m_output_layer_name;
  const std::string m_output_node_name;
  const std::string m_parameter_name;

  std::unique_ptr<const lwt::LightweightGraph> m_graph_even;
  std::unique_ptr<const lwt::LightweightGraph> m_graph_odd;

  NodeMap m_node_map;
  ValueMap *m_val_map;
};

#endif // PARAMETRICNET_H_
