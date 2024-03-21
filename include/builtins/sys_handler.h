#ifndef KIWI_BUILTINS_SYSHANDLER_H
#define KIWI_BUILTINS_SYSHANDLER_H

#include "math/functions.h"
#include "parsing/builtins.h"
#include "parsing/tokens.h"
#include "typing/value.h"
#include "util/sys.h"

class SysBuiltinHandler {
 public:
  static Value execute(const Token& term, const KName& builtin,
                       const std::vector<Value>& args) {
    switch (builtin) {
      case KName::Builtin_Sys_Exec:
        return executeExec(term, args);

      case KName::Builtin_Sys_ExecOut:
        return executeExecOut(term, args);

      case KName::Builtin_Sys_EffectiveUserId:
        return executeEffectiveUserId(term, args);

      default:
        break;
    }

    throw UnknownBuiltinError(term, "");
  }

 private:
  static Value executeEffectiveUserId(const Token& term,
                                      const std::vector<Value>& args) {
    if (args.size() != 0) {
      throw BuiltinUnexpectedArgumentError(term, SysBuiltins.EffectiveUserId);
    }

    return static_cast<k_int>(Sys::getEffectiveUserId());
  }

  static Value executeExec(const Token& term, const std::vector<Value>& args) {
    if (args.size() != 1) {
      throw BuiltinUnexpectedArgumentError(term, SysBuiltins.Exec);
    }

    k_string command = get_string(term, args.at(0));
    return Sys::exec(command);
  }

  static Value executeExecOut(const Token& term,
                              const std::vector<Value>& args) {
    if (args.size() != 1) {
      throw BuiltinUnexpectedArgumentError(term, SysBuiltins.Exec);
    }

    k_string command = get_string(term, args.at(0));
    return Sys::execOut(command);
  }
};

#endif