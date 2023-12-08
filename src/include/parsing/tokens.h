#ifndef TOKENS_H
#define TOKENS_H

#include <variant>
#include <string>

ValueType get_value_type(std::variant<int, double, bool, std::string> v) {
    ValueType type = ValueType::None;
    std::visit([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, int>)
            type = ValueType::Integer;
        else if constexpr (std::is_same_v<T, double>)
            type = ValueType::Double;
        else if constexpr (std::is_same_v<T, bool>)
            type = ValueType::Boolean;
        else if constexpr (std::is_same_v<T, std::string>)
            type = ValueType::String;
        else
            type = ValueType::Unknown;
    }, v);
    return type;
}

enum TokenType {
    IDENTIFIER,
    COMMENT,
    KEYWORD,
    OPERATOR,
    LITERAL,
    STRING,
    NEWLINE,
    ENDOFFILE
};

struct Token {
    TokenType type;
    ValueType value_type;
    std::variant<int, double, bool, std::string> value;

    Token(TokenType t, const std::variant<int, double, bool, std::string>& v) : type(t), value(v) {
        value_type = get_value_type(v);
    }

    std::string toString() {
        if (value_type != ValueType::String)
            throw new std::runtime_error("Value type is not a `String`.");
        return std::get<std::string>(value);
    }

    int toInteger() {
        if (value_type != ValueType::Integer)
            throw new std::runtime_error("Value type is not an `Integer`.");
        return std::get<int>(value);
    }

    bool toBoolean() {
        if (value_type != ValueType::Boolean)
            throw new std::runtime_error("Value type is not a `Boolean`.");
        return std::get<bool>(value);
    }

    double toDouble() {
        if (value_type != ValueType::Double)
            throw new std::runtime_error("Value type is not a `Double`.");
        return std::get<double>(value);
    }
};

#endif