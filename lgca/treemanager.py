class TreeManager:
    def __init__(self):
        self.next_id = 1
        self.tree = {}

    def register(self, parent=None):
        if parent is None:
            self.tree[self.next_id] = {'parent': None, 'origin': None}
        else:
            ori = self.tree[parent]['origin']
            if ori is None:
                ori = parent
            self.tree[self.next_id] = {'parent': parent, 'origin': ori}

        self.next_id += 1
