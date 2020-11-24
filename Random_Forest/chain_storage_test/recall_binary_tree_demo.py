
#先序序列
#[A,B,X,C,X,X,X]

class Node(object):

    def __init__(self, data=None,next_left =None,next_right =None,pre=None):
        self.data = data
        self.pre = pre
        self.next_left = next_left
        self.next_right = next_right

class RBTree(object):

    def __init__(self):

        head = Node('head')
        self.head = head

        self.trunk = None
        self.leaf_l = list()


    def creat_rbtree(self,node_list):

        self.trunk = self.__build_tree(node_list,self.head)
        return self.trunk

    def __build_tree(self,node_list,father_node):
        # self.node_list = node_list #node_list  = ['A','B','#','C','#','#','#']

        # 根节点
        value = node_list.pop(0)
        if value == '#': #
            leaf_node = Node(value,pre= father_node)
            self.leaf_l.append(leaf_node)
            return leaf_node
        else:

            node = Node(value)
            # 左节点
            node.next_left = self.__build_tree(node_list,node)

            # 右节点
            node.next_right = self.__build_tree(node_list,node)
            node.pre = father_node

        return Node(value,node.next_left,node.next_right,father_node)

    def order_btree(self, trunk):
        ##先序
        print(trunk.data)
        if trunk.next_left != None: #若trunk.data == #,表示是叶子节点
            self.order_btree(trunk.next_left)
        if trunk.next_right != None:
            self.order_btree(trunk.next_right)

        #中序
        # if trunk.next_left != None:
        #     self.recall_btree(trunk.next_left)
        # print(trunk.data)
        # if trunk.next_right != None:
        #     self.recall_btree(trunk.next_right)


    def recall_btree(self):
        #
        leaf_num = len(self.leaf_l)

        print('---叶子回溯---- \n')
        for i in range(leaf_num):
            node = self.leaf_l[i]
            print(node.data)
            while node.pre != None:
                print(node.pre.data)
                node = node.pre
            print('-----一个叶子回溯结束--------\n')

if __name__ == '__main__':

    rbt = RBTree()
    node_list  = ['A','B','#','C','#','#','D','#','E','#','#']
    trunk = rbt.creat_rbtree(node_list)
    #遍历树
    rbt.order_btree(trunk)

    #回溯
    rbt.recall_btree()


    # for i in range(7):
    #     ch = node_list.pop(0) #移除第一个元素
    #     print(ch)
    #     print(node_list)


