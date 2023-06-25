from itertools import chain
import math

width = 14
height = 25

class SimpleModel():
    def __init__(self, train_letters, test_letters) -> None:
        # Train letters is a dictionary that maps a character to the correct representation of the character
        # For example: {'a': ['', '', '', ...], 'b': ['', '', '', ...] ...}
        self.train_letters = train_letters

        # List of test character representations for which we need to find the correct character
        # For example: [['', '', '', ...], ['', '', '', ...] ...]
        self.test_letters = test_letters

        # Exist states in our HHM
        self.states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

    @staticmethod
    def convert(matrix):
        converted_matrix = []
        for row in matrix:
            converted_row = [1 if row[i]=="*" else 0 for i in range(len(row))]
            converted_matrix.append(converted_row)
        return converted_matrix

    @staticmethod
    def flatten_to_1D(matrix):
        return list(chain.from_iterable(matrix))

    @staticmethod
    def union_and_intersection(train, test):
        union = 0
        intersection = 0
        for i in range(len(train)):
            union += train[i] or test[i]
            intersection += train[i] and test[i]
        return union, intersection

    def probability(self, training_matrix, test_matrix):
        train = self.flatten_to_1D(self.convert(training_matrix))
        test = self.flatten_to_1D(self.convert(test_matrix))
        union, intersection = self.union_and_intersection(train, test)
        noisy_pixels = union - intersection
        diff = (noisy_pixels + 1) / (height * width)
        return (1-diff)

    def probability_2(self, training_matrix, test_matrix):
        # Intersection over Union (IoU)
        train = self.flatten_to_1D(self.convert(training_matrix))
        test = self.flatten_to_1D(self.convert(test_matrix))
        union, intersection = self.union_and_intersection(train, test)
        probability = ((intersection + 1) / (union + 1))
        return probability

    def probability_3(self, training_matrix, test_matrix):
        # Reference: 
        train = self.flatten_to_1D(self.convert(training_matrix))
        test = self.flatten_to_1D(self.convert(test_matrix))
        union, intersection = self.union_and_intersection(train, test)
        sum_of_train = sum(train)
        sum_of_test = sum(test)
        #score = (intersection + 1) / (max(sum_of_train, sum_of_test) + 1)
        score = (intersection + 1) / (sum_of_train+ sum_of_test - intersection + 1)
        return score

    def pixel_wise_probability(self, train_char_flat, test_char_flat):
        probability = 1
        penalty = 0.2 # m % of noisy data, given in the hints of the problem.
        reward = 1 - penalty
        for i in range(len(test_char_flat)):
            # True positives and true negatives
            result = (train_char_flat[i] and test_char_flat[i]) or (not train_char_flat[i] and not test_char_flat[i])
            if(result):
                probability *= reward
            else:
                probability *= penalty
        return probability

    def simple_model(self, training_matrices, test_matrices):
        res = ""
        for test_matrix in test_matrices:
            max_prob = 0
            argmax_char = ''
            for state in self.states:
                training_matrix = training_matrices[state]
                probability = self.probability_2(training_matrix, test_matrix)
                if probability > max_prob:
                    max_prob = probability
                    argmax_char = state
            res += argmax_char
        return res



class HMM(SimpleModel):
    def __init__(self, train_letters, test_letters, train_txt) -> None:
        super().__init__(train_letters, test_letters)
        # Training data
        self.train_txt = train_txt

        # Dictionary that represents the initial probability table
        # For example: {'a': 0.234, ...}
        self.init_prob = dict()

        # Dictionary of dictionaries that represent the transition probability table
        # Stores the negative logs of probabilities of transitioning from one character to another
        # For example: {'a': {'a': 0.1, 'b': 0.2 ..}, 'b': {'a': 0.3, 'b': 0.4} ...}
        self.trans_prob = dict()

        # Dictionary of dictionaries that represent the emission probability table
        # For example: { 1: {'a': 0.234, 'b': 0.4 ..},  ...}
        self.emit_prob = dict()

        # Observed sequence
        self.obs_sequence = []

    
    def convert(self, character_matrix):
        return SimpleModel.convert(character_matrix)
    
    def flatten_to_1D(self, sparse_matrix):
        return SimpleModel.flatten_to_1D(sparse_matrix)
    
    def probability_2(self, training_matrix, test_matrix):
        return super().probability_2(training_matrix, test_matrix)
    
    def probability_3(self, training_matrix, test_matrix):
        return super().probability_3(training_matrix, test_matrix)
    
    def pixel_wise_probability(self, train_char_flat, test_char_flat):
        return super().pixel_wise_probability(train_char_flat, test_char_flat)

    
    def cal_emission(self, train_matrices, test_matrices):
        # Calculating emission probabilities, representing the likelihood of observing a test character given a hidden character.
        n = len(test_matrices)
        obs_sequence = ['observed_'+str(i) for i in range(n)]
        emission_table = {state:{observed:0 for observed in obs_sequence} for state in self.states}

        for i in range(n):
            for state in self.states:
                emission_table[state][obs_sequence[i]] = math.log(self.probability_3(train_matrices[state], test_matrices[i])) * 50
        
        return emission_table
    
    
    def initial_transition_train(self):
        # Calculating initial probabilities for each hidden state, representing the probability of starting with each character.
        raw_word_data = []
        training_file_path = self.train_txt
        training_file = open(training_file_path, 'r')
        for line in training_file:
            word_and_pos = tuple([word for word in line.split()])
            # For bc.train where there are word and POS in alternative occurences, use the below linw.
            raw_word_data += [(word_and_pos[0::2]), ]\

        cleaned_data = ""
        for line1 in raw_word_data:
            new_sentence = ""
            for line in line1:
                new_sentence += " " + ''.join(char for char in line if char in self.states)
            new_sentence = new_sentence.replace(' ,', ',')
            new_sentence = new_sentence.replace(' ,', ',')
            new_sentence = new_sentence.replace(' \'\'','\"')
            new_sentence = new_sentence.replace('`` ','\"')
            new_sentence = new_sentence.replace('``','\"')
            new_sentence = new_sentence.replace("  ", " ")
            new_sentence = new_sentence.replace(" .", ".").strip()
            cleaned_data += new_sentence + "\n"
        
        cleaned_data = cleaned_data.strip()
        char_prob = dict()
        for char in self.states:
            char_prob[char] = 0
        transition_frequencies = {i:{j:0.1 for j in self.states} for i in self.states}
        for i in range(len(cleaned_data) - 1):
            cur_char = cleaned_data[i]
            next_char = cleaned_data[i+1]
            
            if(cur_char in transition_frequencies):
                char_prob[cur_char] += 1
                if(next_char in transition_frequencies[cur_char]):
                    transition_frequencies[cur_char][next_char] += 1

        total_log = math.log(sum(char_prob.values()))
        for key, val in char_prob.items():
            char_prob[key] = 10000000.0 if val < 1 else total_log - math.log(val)

        # Calculating transition probabilities between hidden states, representing the likelihood of transitioning between characters.
        initial_frequencies = {i: 0.00000000001 for i in self.states}
        for char in cleaned_data:
            if(char in initial_frequencies):
                initial_frequencies[char] += 1
        # Too many spaces and quotes in the dataset. Also, in reality, initial character being a space or quote is quite low.
        initial_frequencies[' '] = initial_frequencies[' '] / 1000
        initial_frequencies["'"] = initial_frequencies["'"] / 10
        initial_probabilities = {item: math.log(value/sum(initial_frequencies.values())) for item, value in initial_frequencies.items()}
        
        # Summing up probabilities in a dictionary comprehension
        # https://stackoverflow.com/questions/30964577/divide-each-python-dictionary-value-by-total-value/30964739
        transition_probabilities = {i:{k: math.log(v/total) for total in (sum(transition_frequencies[i].values()),) for k, v in transition_frequencies[i].items()} for i in self.states}

        return initial_probabilities, transition_probabilities


    def perform_ocr_forward(self, n):
        """
        Performs Optical Character Recognition (OCR) using the forward algorithm on a Hidden Markov Model (HMM).

        Args:
            observed (str): The sequence of test characters to be recognized.
            states (list): The list of possible hidden states (characters).
            initial (dict): The initial probabilities for each hidden state.
            emission (dict): The emission probabilities representing the likelihood of observing a test character given a hidden character.
            transition (dict): The transition probabilities representing the likelihood of transitioning between hidden states.
            n (int): The length of the observed sequence.

        Returns:
            str: The predicted sequence of hidden states (characters) based on the forward algorithm.
        """

        forward_probabilities = {state: [0] * n for state in self.states}
        predicted_sequence = [""] * n

        # Initialize the forward probabilities at the first time step
        for state in self.states:
            forward_probabilities[state][0] = self.init_prob[state] * self.emit_prob[state][self.obs_sequence[0]]

        # Iterate through each observed state and update the forward probabilities
        for t in range(1, n):
            for state in self.states:
                forward_probabilities[state][t] = sum(
                    forward_probabilities[prev_state][t - 1] * self.trans_prob[prev_state].get(state, 0) * self.emit_prob[state][self.obs_sequence[t]]
                    for prev_state in self.states
                )

        # Calculate the predicted sequence of hidden states using Viterbi decoding
        predicted_sequence[n - 1] = max(forward_probabilities, key=forward_probabilities.get)

        for t in range(n - 2, -1, -1):
            max_state = None
            max_prob = float('-inf')
            for state in self.states:
                if predicted_sequence[t + 1] in self.trans_prob[state]:
                    prob = forward_probabilities[state][t] * self.trans_prob[state][predicted_sequence[t + 1]]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state
            if max_state is None:
                max_state = self.states[0]  # Use a fallback state
            predicted_sequence[t] = max_state

        return "".join(predicted_sequence)
    


    def viterbi(self, n):
        """
        Performs Optical Character Recognition (OCR) using the Viterbi algorithm on a Hidden Markov Model (HMM).

        Args:
            test_characters (str): The sequence of test characters to be recognized.

        Returns:
            str: The best-matching string of characters based on the Viterbi algorithm.

        The function constructs an HMM where each hidden state represents a character and each observed state represents a
        test character. Transition probabilities between hidden states represent the likelihood of transitioning from one
        character to another, while emission probabilities represent the likelihood of observing a test character given a
        hidden character.

        The Viterbi algorithm is then used to find the most likely sequence of hidden states given the observed states. The
        steps involved in the process include:

        - Initializing the Viterbi trellis, which stores the probabilities of the most likely sequence of hidden states at each time step.
        - Iterating through each observed state (test character) and updating the Viterbi trellis to find the most likely hidden state at each time step.
        - Backtracking through the Viterbi trellis to find the most likely sequence of hidden states.
        - Returning the sequence of hidden states (characters) as the OCR result.
        """

        # Implementation goes here
        viterbi_table = {state:[0] * n for state in self.states}
        which_table = {state:[0] * n for state in self.states}
        for state in self.states:
            viterbi_table[state][0] =  self.init_prob[state] + self.emit_prob[state][self.obs_sequence[0]]
        for i in range(1, n):
            for state in self.states:
                (which_table[state][i], viterbi_table[state][i]) =  max([(s0, viterbi_table[s0][i-1] + self.trans_prob[s0][state]) for s0 in self.states], key=lambda l:l[1]) 
                viterbi_table[state][i] += self.emit_prob[state][self.obs_sequence[i]]
        
        viterbi_seq = [""] * n
        temp = [(state, viterbi_table[state][n-1]) for state in self.states]
        viterbi_seq[n-1], _ = max(temp, key=lambda array:array[1])
        for i in range(n-2, -1, -1):
            viterbi_seq[i] = which_table[viterbi_seq[i+1]][i+1]
        
        return ''.join(viterbi_seq)

    def hidden_markov_model(self, train_matrices, test_matrices):
        self.init_prob, self.trans_prob = self.initial_transition_train()
        self.emit_prob = self.cal_emission(train_matrices, test_matrices)
        n = len(test_matrices)
        self.obs_sequence = ['observed_' + str(i) for i in range(n)]
        return self.perform_ocr_forward(n), self.viterbi(n)


