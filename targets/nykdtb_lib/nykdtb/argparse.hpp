#ifndef NYKDTB_ARGPARSE_HPP
#define NYKDTB_ARGPARSE_HPP

#include "nykdtb/types.hpp"

namespace nykdtb {

class ArgumentParser final {
public:
    NYKDTB_DEFINE_EXCEPTION_CLASS(IncorrectParameterType, RuntimeException)
    NYKDTB_DEFINE_EXCEPTION_CLASS(NotAllArgumentsParsed, LogicException)

public:
    using ArgumentList = Vec<std::string>;

    struct Elem {
        enum class Type { Parameter, SwitchOneLetter, Switch, Invalid, End };
        Type type;
        std::string value;

        Elem(std::string input);
        inline Elem()
            : type(Type::Invalid) {}
        inline Elem(Type value)
            : type(value) {}
        inline Elem(Type t, std::string input)
            : type(t), value(input) {}
        inline bool operator==(const Elem& rhs) const { return type == rhs.type && value == rhs.value; }
        inline bool operator==(const Elem::Type& rhs) const { return type == rhs; }
    };

public:
    ArgumentParser(int argc, char* argv[]);
    ~ArgumentParser() noexcept(false);

    Size getRemainingArgumentsCount() { return m_remainingArguments.size(); }
    Size getAllArgumentsCount() { return m_allArguments.size(); }

    Elem parseNextArgument();
    Elem parseNextArgument(const Elem::Type expectedType);

    static std::string fileExtension(const std::string& input);

private:
    ArgumentList m_allArguments;
    ArgumentList m_remainingArguments;
};

using AP   = ArgumentParser;
using APE  = ArgumentParser::Elem;
using APET = ArgumentParser::Elem::Type;

}  // namespace nykdtb

#endif