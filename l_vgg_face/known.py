import pickle


class KnownPerson:
    def __init__(self):
        self.known_list = []

    def add_person(self, name, features):
        self.known_list.append({
            'name': name,
            'features': features
        })

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            known_person = pickle.load(f)
        return known_person
