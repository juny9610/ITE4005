import sys
import math
import pandas as pd

'''
Node class
Decision tree의 node 생성한다.
'''
class Node:
    def __init__(self, attribute, is_leaf=False, label_name=None):
        self.attribute = attribute
        self.child = dict()
        self.is_leaf = is_leaf
        self.label_name = label_name

'''
Decision tree class
Decision tree의 구조와 이를 구현하는데 필요한 함수가 선언되어있다.
'''
class DecisionTree:
    def __init__(self, df, method):
        self.method = method
        self.label = df.columns[-1]
        self.attri_dict = {attribute: df[attribute].unique() 
                            for attribute in df.columns if attribute!=self.label}
        self.root =  self.create_decision_tree(df)

    '''
    Calculate entropy
    주어진 data의 entropy를 구한다.
    '''
    def entropy(self, df):
        ## Info(D) = -Σ(p(i)*log2(p(i))) i:1~m (m은 class labels의 수)
        e = .0
        for _, value in df[self.label].value_counts().iteritems():
            p_value = value / len(df)
            ## log2(0)의 오류를 피하기 위해 1e-9를 더했다.
            e += -1.0 * (p_value * math.log(p_value + 1e-9, 2))
        return e

    '''
    Calculate information gain
    주어진 data의 information gain을 구한다.
    '''
    def info_gain(self, df, attribute):
        ## InfoA(D) = Σ(|D(j)|/|D|*Info(D(j)) (|D|는 df의 크기)
        e = .0
        info_D = self.entropy(df)
        for a in self.attri_dict[attribute]:
            filtered_df = df[df[attribute] == a]
            info_A = self.entropy(filtered_df)
            e += info_A * (len(filtered_df) / len(df))

        return info_D - e

    '''
    Calculate split information
    주어진 data의 split information을 구한다.
    '''
    def split_info(self, df, attribute):
        ## SplitInfoA(D) = -Σ(|D(j)|/|D|*log2(|D(j)|/|D|)) (|D|는 df의 크기)
        e = .0
        for a in self.attri_dict[attribute]:
            filtered_df = df[df[attribute] == a]
            p_value = len(filtered_df) / len(df)
            value = math.log(p_value + 1e-9, 2)
            ## log2(0)의 오류를 피하기 위해 1e-9를 더했다.
            e += -1.0 * (p_value * value)

        return e
    
    '''
    Calculate gain ratio
    주어진 data의 gain ratio를 구한다.
    '''
    def gain_ratio(self, df, attribute):
        ## GainRatio(A) = Gain(A)/SplitInfo(A)
        gain = self.info_gain(df,attribute)
        split_info = self.split_info(df,attribute)

        return gain / split_info

    '''
    Calculate gini
    주어진 data의 gini를 구한다.
    '''
    def gini(self, df):
        ## gini(D) = 1-Σp(j)^2
        e = 1.0
        for _, value in df[self.label].value_counts().iteritems():
            p = value / len(df)
            e -= p ** 2

        return e

    '''
    Calculate gini index
    주어진 data의 gini index를 구한다.
    '''
    # Calculate GiniA(D)
    def gini_index(self, df, attribute, left, right):
        ## giniA(D) = |D1|gini(D1)+|D2|gini(D2) (여기서 D1은 left subset, D2는 right subset)
        ## Left subset
        left_df = df[df[attribute].isin(left)]
        left_size = len(left_df) / len(df)
        left_gini = self.gini(left_df)
        # Right subset
        right_df = df[df[attribute].isin(right)]
        right_size = len(right_df) / len(df)
        right_gini = self.gini(right_df)

        return (left_size * left_gini) + (right_size * right_gini)

    '''
    Create decision tree
    주어진 data로 decision tree를 생성한다.
    '''
    def create_decision_tree(self, df):
        ## Data frame의 classification이 잘 되었을 때, leaf node를 return한다.
        if df[self.label].nunique() == 1:
            label_name = df[self.label].unique()[0]
            return Node(None, True, label_name)

        ## Data frame의 classification이 다 되지 않았을 때, majority voting을 진행한다.
        elif len(df.columns) == 1:
            ## majority voting
            majority_list = df_train[self.label].value_counts().sort_values(ascending=False)
            majority = majority_list.index[0]
            return Node(None, True, majority)

        ## method에 따라 attribute를 결정한 후, node를 생성한다.
        ## method == 0 : Information gain
        if self.method == 0:
            info_dict = {attribute: self.info_gain(df,attribute) 
                            for attribute in df.columns if attribute!=self.label}
            target_attri = sorted(info_dict.items(), key=lambda x: x[1], reverse=True)[0][0]

        ## method == 1 : Gain ratio 
        elif self.method == 1:
            ratio_dict = {attribute: self.gain_ratio(df,attribute) 
                            for attribute in df.columns if attribute!=self.label}
            target_attri = sorted(ratio_dict.items(), key=lambda x: x[1], reverse=True)[0][0]        

        node = Node(target_attri)

        ## data의 수가 가장 많은 label을 가져온다.
        majority_list = df_train[self.label].value_counts().sort_values(ascending=False)
        node.label_name = majority_list.index[0]

        ## gini index를 이용하여 적절한 branch를 결정한다.
        a = self.attri_dict[node.attribute]
        gini_dict = dict()
        for i in range(1,len(a)):
            left = tuple(a[:i])
            right = tuple(a[i:])
            gini_dict[(left,right)] = self.gini_index(df, node.attribute, left, right)
        branch = sorted(gini_dict.items(),key=lambda x: x[1], reverse=False)[0][0]
        for b in branch:
            filtered_df = df[df[node.attribute].isin(b)]
            if len(filtered_df) > 0:
                node.child[tuple(b)] = self.create_decision_tree(filtered_df)
            else:
                majority_list = df_train[self.label].value_counts().sort_values(ascending=False)
                majority = majority_list.index[0]
                node.child[tuple(b)] = Node(None,True,majority) 
        return node

    '''
    Find leaf
    주어진 data의 leaf node를 찾는다.
    '''
    def find_leaf(self, data):
        node = self.root
        a = data[node.attribute]
        while not node.is_leaf:
            for child, next_node in node.child.items():
                if a in child:
                    break
            node = next_node
            if node.attribute:
                a = data[node.attribute]

        return node.label_name

    '''
    Classify
    주어진 data를 classify 한다.
    '''
    def classify(self, df):
        label = [self.find_leaf(df.loc[i]) for i in range(len(df))]
        df[self.label] = label
        return df

'''
Main 함수
'''
if __name__ == "__main__":
    ## read argv
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    ## read file
    df_train = pd.read_csv(train_file, sep="\t")
    df_test = pd.read_csv(test_file, sep="\t")

    ## Build decision tree
    dt = DecisionTree(df_train, 0)
    
    ## Classify data
    df = dt.classify(df_test)
    df.to_csv(output_file, index=False, sep="\t")