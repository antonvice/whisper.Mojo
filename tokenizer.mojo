from collections import List


struct Tokenizer:
    var vocab: List[String]

    fn __init__(out self, path: String) raises:
        self.vocab = List[String]()
        with open(path, "r") as f:
            var content = f.read()
            var tokens = content.split("\n")
            for i in range(len(tokens)):
                self.vocab.append(String(tokens[i]))

    fn decode(self, tokens: List[Int]) -> String:
        var result: String = ""
        for i in range(len(tokens)):
            var token_id = tokens[i]
            if token_id >= 0 and token_id < len(self.vocab):
                var token = self.vocab[token_id]
                # Filter out special tokens (they usually look like <|...|>)
                if not (token.startswith("<|") and token.endswith("|>")):
                    # Whisper uses Gpt-2 style BPE where "Ġ" is a space
                    var clean_token = token.replace("Ġ", " ")
                    # Replace literal \n with actual newline if needed
                    clean_token = clean_token.replace("\\n", "\n")
                    result += clean_token
        return result
