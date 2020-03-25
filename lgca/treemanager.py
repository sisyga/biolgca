class TreeManager:
    def __init__(self):
        self.next_id = 1
        self.tree = {}

    def register(self, parent=None):
        if parent is None:
            self.tree[self.next_id] = {'parent': self.next_id, 'origin': self.next_id}
        else:
            ori = self.tree[parent]['origin']
            self.tree[self.next_id] = {'parent': parent, 'origin': ori}

        self.next_id += 1




