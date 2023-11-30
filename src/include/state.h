#ifndef NOCTIS_STATE_H
#define NOCTIS_STATE_H

struct
{
    std::string InitialDirectory;
    bool CaptureParse;
    std::string ParsedOutput;
    std::string CurrentLine;
    std::string CurrentMethodClass;
    std::string CurrentModule;
    std::string CurrentClass;
    std::string CurrentScript;
    std::string CurrentScriptName;
    std::string GoTo;
    std::string LastValue;
    std::string LogFile;
    std::string Application;
    std::string PreviousScript;
    std::string PromptStyle;
    std::string SwitchVarName;
    std::string DefaultLoopSymbol;
    std::string Null;
    double NullNum;
    int ArgumentCount;
    int BadMethodCount;
    int BadClassCount;
    int BadVarCount;
    int CurrentLineNumber;
    int IfStatementCount;
    int ForLoopCount;
    int ParamVarCount;
    int WhileLoopCount;

    bool IsCommented;
    bool IsMultilineComment;

    bool Breaking;
    bool DefiningIfStatement;
    bool DefiningForLoop;
    bool DefiningLocalForLoop;
    bool DefiningLocalSwitchBlock;
    bool DefiningLocalWhileLoop;
    bool DefiningMethod;
    bool DefiningModule;
    bool DefiningNest;
    bool DefiningClass;
    bool DefiningClassMethod;
    bool DefiningParameterizedMethod;
    bool DefiningPrivateCode;
    bool DefiningPublicCode;
    bool DefiningScript;
    bool DefiningSwitchBlock;
    bool DefiningWhileLoop;
    bool DontCollectMethodVars;
    bool ExecutedIfStatement;
    bool ExecutedMethod;
    bool ExecutedTemplate;
    bool FailedIfStatement;
    bool FailedNest;
    bool GoToLabel;
    bool InDefaultCase;
    bool Returning;
    bool UseCustomPrompt;

    std::string ErrorVarName;
    std::string LastError;
    int LastErrorCode;
    bool ExecutedTryBlock;
    bool RaiseCatchBlock;
    bool SkipCatchBlock;
    bool SuccessFlag;
} State;

#endif
