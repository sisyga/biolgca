class TreeManager:
    def __init__(self):
        self.next_id = 1
        self.next_fam = 1
        self.tree = {}

    def register(self, cell=None): #cell = label der Mutterzelle
        if cell is None:
            self.tree[self.next_id] = {'family': self.next_fam, 'parent': self.next_fam, 'origin': self.next_fam}
            self.next_fam += 1
        else:
            fam = self.tree[cell]['family']
            par = self.tree[cell]['parent']
            ori = self.tree[cell]['origin']
            self.tree[self.next_id] = {'family': fam, 'parent': par, 'origin': ori}
        self.next_id += 1

    def register_mutation(self, parentcell):
        ori = self.tree[parentcell]['origin']
        par = self.tree[parentcell]['family']
        self.tree[self.next_id] = {'family': self.next_fam, 'parent': par, 'origin': ori}
        self.next_fam += 1
        self.next_id += 1



