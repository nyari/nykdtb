#include "nykdtb/argparse.hpp"

#include <regex>

namespace nykdtb {

ArgumentParser::Elem::Elem(std::string input) {
    type = Type::Invalid;
    if (input.size() == 1) {
        type  = Type::Parameter;
        value = mmove(input);
    } else if (input.size() >= 2) {
        if (input[0] == '-') {
            if (input[1] == '-') {
                if (input.size() == 2) {
                    type  = Type::Parameter;
                    value = mmove(input);
                } else {
                    type  = Type::Switch;
                    value = std::string(input.begin() + 2, input.end());
                }
            } else {
                // Negative number handling
                if ((('0' <= input[1]) && (input[1] <= '9')) || (input[1] == '.')) {
                    type  = Type::Parameter;
                    value = std::string(input);
                } else {
                    type  = Type::SwitchOneLetter;
                    value = std::string(input.begin() + 1, input.end());
                }
            }
        } else {
            type  = Type::Parameter;
            value = mmove(input);
        }
    }
}

ArgumentParser::ArgumentParser(int argc, char* argv[]) {
    for (int argumentIndex = 0; argumentIndex < argc; ++argumentIndex) {
        m_allArguments.push_back(std::string(argv[argumentIndex]));
    }
    m_remainingArguments = ArgumentList(m_allArguments.begin() + 1, m_allArguments.end());
}

ArgumentParser::Elem ArgumentParser::parseNextArgument() {
    if (m_remainingArguments.empty()) {
        return Elem(Elem::Type::End);
    }

    Elem result(mmove(m_remainingArguments.front()));

    m_remainingArguments.erase(m_remainingArguments.begin());

    return result;
}

ArgumentParser::Elem ArgumentParser::parseNextArgument(const Elem::Type expectedType) {
    if (m_remainingArguments.empty()) {
        if (expectedType != Elem::Type::End) {
            throw IncorrectParameterType();
        } else {
            return Elem(Elem::Type::End);
        }
    }

    Elem result(m_remainingArguments.front());
    if (result.type != expectedType) {
        throw IncorrectParameterType();
    }

    m_remainingArguments.erase(m_remainingArguments.begin());

    return result;
}

std::string ArgumentParser::fileExtension(const std::string& input) {
    auto lastPeriod = input.end();
    for (auto it = input.begin(); it != input.end(); ++it) {
        if (*it == '.') {
            lastPeriod = it;
        }
    }

    if (lastPeriod == input.end()) {
        return {};
    } else {
        return {lastPeriod + 1, input.end()};
    }
}

}  // namespace nykdtb
