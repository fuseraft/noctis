#ifndef KIWI_BUILTINS_ML_H
#define KIWI_BUILTINS_ML_H

#include <cstdlib>
#include <string>
#include <unordered_map>
#include "math/functions.h"
#include "parsing/builtins.h"
#include "parsing/tokens.h"
#include "typing/value.h"
#include "util/string.h"

const double HALF = 0.5;
const double ONE_PERCENT = 0.01;

class MLBuiltinHandler {
 public:
  static k_value execute(const Token& term, const KName& builtin,
                         const std::vector<k_value>& args) {
    switch (builtin) {
      case KName::Builtin_MLReg_Dropout:
        return executeRegDropout(term, args);

      case KName::Builtin_MLReg_WeightDecay:
        return {};

      case KName::Builtin_MLReg_L1Lasso:
        return {};

      case KName::Builtin_MLReg_L2Ridge:
        return {};

      case KName::Builtin_MLReg_ElasticNet:
        return {};

      default:
        break;
    }

    throw UnknownBuiltinError(term, "");
  }

 private:
  static k_value executeRegDropout(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegDropout);
    }

    k_value dropoutRate = HALF;

    if (args.size() > 1) {
      dropoutRate = args.at(1);
    }

    return MLRegularizationBuiltins.__dropout__(term, args.at(0), dropoutRate);
  }

  static k_value executeRegWeightDecay(const Token& term,
                                       const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value decayRate = ONE_PERCENT;

    if (args.size() > 1) {
      decayRate = args.at(1);
    }

    MLRegularizationBuiltins.__weight_decay__(term, args.at(0), decayRate);
    return args.at(0);
  }

  static k_value executeRegL1Lasso(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value lambda = ONE_PERCENT;

    if (args.size() > 1) {
      lambda = args.at(1);
    }

    return MLRegularizationBuiltins.__l1_regularization__(term, args.at(0),
                                                          lambda);
  }

  static k_value executeRegL2Ridge(const Token& term,
                                   const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 2) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value lambda = ONE_PERCENT;

    if (args.size() > 1) {
      lambda = args.at(1);
    }

    return MLRegularizationBuiltins.__l2_regularization__(term, args.at(0),
                                                          lambda);
  }

  static k_value executeRegElasticNet(const Token& term,
                                      const std::vector<k_value>& args) {
    if (args.empty() || args.size() > 3) {
      throw BuiltinUnexpectedArgumentError(term, MLBuiltins.RegWeightDecay);
    }

    k_value lambda1 = ONE_PERCENT;
    k_value lambda2 = ONE_PERCENT;

    if (args.size() > 1) {
      lambda1 = args.at(1);
    }
    if (args.size() > 2) {
      lambda2 = args.at(2);
    }

    return MLRegularizationBuiltins.__elastic_net__(term, args.at(0), lambda1,
                                                    lambda2);
  }
};

#endif