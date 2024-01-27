#ifndef ERROR_H
#define ERROR_H

#include <iostream>
#include <exception>
#include "../parsing/tokens.h"

class UslangError : public std::exception {
 public:
  UslangError(Token token, std::string error, std::string message = "")
      : token(token), error(error), message(message) {}

  const char* what() const noexcept override { return message.c_str(); }

  const Token getToken() const { return token; }

  const std::string getError() const { return error; }

  const std::string getMessage() const { return message; }

 private:
  Token token;
  std::string error;
  std::string message;
};

class SyntaxError : public UslangError {
 public:
  SyntaxError(const Token& token, std::string message = "Invalid syntax.")
      : UslangError(token, "SyntaxError", message) {}
};

class ParameterMissingError : public UslangError {
 public:
  ParameterMissingError(const Token& token, std::string name)
      : UslangError(token, "ParameterMissingError",
                    "The parameter `" + name + "` was expected but missing.") {}
};

class ConversionError : public UslangError {
 public:
  ConversionError(const Token& token)
      : UslangError(token, "ConversionError", "A conversion error occurred.") {}
};

class DivideByZeroError : public UslangError {
 public:
  DivideByZeroError(const Token& token)
      : UslangError(token, "DivideByZeroError",
                    "Attempted to divide by zero.") {}
};

class VariableUndefinedError : public UslangError {
 public:
  VariableUndefinedError(const Token& token, std::string name)
      : UslangError(token, "VariableUndefinedError",
                    "Variable `" + name + "` is undefined.") {}
};

// TODO: refine this.
class IllegalNameError : public UslangError {
 public:
  IllegalNameError(const Token& token, std::string name)
      : UslangError(token, "IllegalNameError",
                    "The name `" + name + "` is illegal.") {}
};

class FileNotFoundError : public UslangError {
 public:
  FileNotFoundError(std::string path)
      : UslangError(Token::createEmpty(), "FileNotFoundError",
                    "File not found: " + path) {}

  FileNotFoundError(const Token& token, std::string path)
      : UslangError(token, "FileNotFoundError", "File not found: " + path) {}
};

#endif