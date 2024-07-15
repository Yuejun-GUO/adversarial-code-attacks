import pdb
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import copy
from .run_parser import get_example_batch


python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double",
                    "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String",
                    "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet",
                    "HashMap"]
c_keywords = ["auto", "break", "case", "char", "const", "continue",
              "default", "do", "double", "else", "enum", "extern",
              "float", "for", "goto", "if", "inline", "int", "long",
              "register", "restrict", "return", "short", "signed",
              "sizeof", "static", "struct", "switch", "typedef",
              "union", "unsigned", "void", "volatile", "while",
              "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
              "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
              "_Thread_local", "__func__"]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
            "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
            "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]  # <stdlib.h> macro
c_special_ids = ["main",  # main function
                 "stdio", "cstdio", "stdio.h",  # <stdio.h> & <cstdio>
                 "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",  # <stdio.h> types & streams
                 "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush",  # <stdio.h> functions
                 "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                 "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                 "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                 "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                 "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                 "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                                                                   "stdlib", "cstdlib", "stdlib.h",
                 # <stdlib.h> & <cstdlib>
                 "size_t", "div_t", "ldiv_t", "lldiv_t",  # <stdlib.h> types
                 "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                 "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                 "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                 "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                 "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                 "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                 "mbstowcs", "wcstombs",
                 "string", "cstring", "string.h",  # <string.h> & <cstring>
                 "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",  # <string.h> functions
                 "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                 "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                 "strpbrk", "strstr", "strtok", "strxfrm",
                 "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",  # <string.h> extension functions
                 "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                 "iostream", "istream", "ostream", "fstream", "sstream",  # <iostream> family
                 "iomanip", "iosfwd",
                 "ios", "wios", "streamoff", "streampos", "wstreampos",  # <iostream> types
                 "streamsize", "cout", "cerr", "clog", "cin",
                 "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",  # <iostream> manipulators
                 "noshowbase", "showpoint", "noshowpoint", "showpos",
                 "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                 "left", "right", "internal", "dec", "oct", "hex", "fixed",
                 "scientific", "hexfloat", "defaultfloat", "width", "fill",
                 "precision", "endl", "ends", "flush", "ws", "showpoint",
                 "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",  # <math.h> functions
                 "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                 "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                 "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']


def select_parents(population):
    length = range(len(population))
    index_1 = random.choice(length)
    index_2 = random.choice(length)
    chromesome_1 = population[index_1]
    chromesome_2 = population[index_2]
    return chromesome_1, index_1, chromesome_2, index_2


def mutate(chromesome, variable_substitue_dict):
    tgt_index = random.choice(range(len(chromesome)))
    tgt_word = list(chromesome.keys())[tgt_index]
    chromesome[tgt_word] = random.choice(variable_substitue_dict[tgt_word])

    return chromesome


def crossover(csome_1, csome_2, r=None):
    if r is None:
        r = random.choice(range(len(csome_1)))  # 随机选择一个位置.
        # 但是不能选到0

    child_1 = {}
    child_2 = {}
    for index, variable_name in enumerate(csome_1.keys()):
        if index < r:  # 前半段
            child_2[variable_name] = csome_1[variable_name]
            child_1[variable_name] = csome_2[variable_name]
        else:
            child_1[variable_name] = csome_1[variable_name]
            child_2[variable_name] = csome_2[variable_name]
    return child_1, child_2


def map_chromesome(chromesome: dict, code: str, lang: str) -> str:
    temp_replace = get_example_batch(code, chromesome, lang)

    return temp_replace


from keyword import iskeyword


def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)


def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True


def is_valid_variable_c(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True


def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False


def is_valid_substitue(substitute: str, tgt_word: str, lang: str) -> bool:
    '''
    determine the validity of substitues
    '''
    is_valid = True

    if not is_valid_variable_name(substitute, lang):
        is_valid = False

    return is_valid


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    '''
    得到substitues
    '''
    device = next(mlm_model.parameters()).device
    # substitutes L, k
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes[:24]:  # 去掉不用的计算.
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # 不是，这个总共不会超过24... 那之前生成那么多也没用....
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    '''
    将生成的substitued subwords转化为words
    '''
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        # 比如空格对应的subwords就是[a,a]，长度为0
        return words

    elif sub_len == 1:
        # subwords就是本身
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._decode([int(i)]))
            # 将id转为token.
    else:
        # word被分解成了多个subwords
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_masked_code_by_position(tokens: list, positions: dict):
    '''
    given a code and the position needed to be masked，return the masked code
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [<mask>, b, c]
            [a, b, <mask>]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['<unk>'] + tokens[pos + 1:])
            replace_token_positions.append(pos)

    return masked_token_list, replace_token_positions


def get_masked_code_by_positions(tokens: list, positions: dict):
    '''
    given a code and the positions to mask, return the masked code
    Example:
        tokens: [a,b,a,c]
        positions: [0,2]
        Return:
            [<mask>, b, <mask>, c]
    '''
    masked_token_list = []
    for variable_name in positions.keys():
        tmp = copy.deepcopy(tokens)
        for pos in positions[variable_name]:
            tmp[pos] = '<unk>'
        masked_token_list.append(tmp)

    return masked_token_list


__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]
__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
           ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
           "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
           ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]  # <stdlib.h> macro
__special_ids__ = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",  # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",  # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush",  # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                                                                     "stdlib", "cstdlib", "stdlib.h",
                   # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",  # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",  # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",  # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk", "strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",  # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",  # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",  # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",  # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",  # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]


def isUID(_text=""):
    '''
    Return if a token is a UID.
    '''

    _text = _text.strip()
    if _text == '':
        return False

    if " " in _text or "\n" in _text or "\r" in _text:
        return False
    elif _text in __key_words__:
        return False
    elif _text in __ops__:
        return False
    elif _text in __macros__:
        return False
    elif _text in __special_ids__:
        return False
    elif _text[0].lower() in "0123456789":
        return False
    elif "'" in _text or '"' in _text:
        return False
    elif _text[0].lower() in "abcdefghijklmnopqrstuvwxyz_":
        for _c in _text[1:-1]:
            if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_":
                return False
    else:
        return False
    return True


def getUID(_tokens=[], uids=[]):
    '''
    Return all UIDs and their indeces, given a token sequence.
    '''

    ids = {}
    for i, t in enumerate(_tokens):
        if isUID(t) and t in uids[0].keys():
            if t in ids.keys():
                ids[t].append(i)
            else:
                ids[t] = [i]
    return ids


def get_importance_score(words_list: list, variable_names: list, tgt_model, tokenizer, label, ori_prob, device):
    """Compute the importance score of each variable"""
    # 1. filter all keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, None

    importance_score = []
    # 2. get Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_feature = tokenizer([new_code], return_tensors="pt", truncation=True, padding='max_length').to(device)
        logits = tgt_model(**new_feature).logits
        logits = F.sigmoid(logits)
        logits = torch.Tensor.cpu(logits).detach().numpy()[0]
        prob = logits[label]
        importance_score.append(ori_prob - prob)

    return importance_score, replace_token_positions, positions, len(importance_score)


def get_changed_code_by_position(tokens: list, positions: dict):
    '''
    given a code and the position to be removed，return the updated code
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [b, c]
            [a, b]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + tokens[pos + 1:])
            replace_token_positions.append(pos)

    return masked_token_list, replace_token_positions


def get_importance_score_fooler(words_list: list, variable_names: list, tgt_model, tokenizer, label, ori_logits, device):
    """Compute the importance score of each variable"""
    # 1. filter all keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, None

    importance_score = []
    # 2. get deleted_tokens
    deleted_token_list, replace_token_positions = get_changed_code_by_position(words_list, positions)
    for index, tokens in enumerate([words_list] + deleted_token_list):
        new_code = ' '.join(tokens)
        new_feature = tokenizer([new_code], return_tensors="pt", truncation=True, padding='max_length').to(device)
        logits = tgt_model(**new_feature).logits
        logits = F.sigmoid(logits)
        logits = torch.Tensor.cpu(logits).detach().numpy()[0]
        pred_label = np.argmax(logits)
        imp_score = ori_logits[label] - logits[label] if pred_label==label else ori_logits[label] - logits[label] + (logits[pred_label] - ori_logits[pred_label])
        importance_score.append(imp_score)

    return importance_score, replace_token_positions, positions, len(importance_score)
