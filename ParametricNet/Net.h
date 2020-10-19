#ifndef NET_H_
#define NET_H_

#include <string>
#include <map>
#include <memory>

// Forward declaration
namespace lwt {
  class LightweightGraph;
}

class Net {

public:
  enum class Fold {Even, Odd};

  Net();
  ~Net();

  // Set the lwtnn network configuration
  // filename_even: Network trained on even event numbers
  // filename_odd: Network trained on odd event numbers
  void init(const std::string &filename_even,
            const std::string &filename_odd);
  void reset();
  void set_variable(const std::string &name, const double val);
  float evaluate(Fold fold);

private:
  using ValueMap = std::map<std::string, double>;
  using NodeMap = std::map<std::string, ValueMap>;

  const std::string m_input_layer_name;
  const std::string m_output_layer_name;
  const std::string m_output_node_name;

  std::unique_ptr<const lwt::LightweightGraph> m_graph_even;
  std::unique_ptr<const lwt::LightweightGraph> m_graph_odd;

  NodeMap m_node_map;
  ValueMap *m_val_map;
};

#endif // NET_H_
