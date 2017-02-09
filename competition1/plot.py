import matplotlib.pyplot as plt

def plot_accs(train_accs, test_accs, indices, label):
    # make teh graph
    plt.figure()
    colors = ['crimson', 'darkcyan']
    plt.plot(indices, train_accs, marker='o', color='crimson', label='train')
    plt.plot(indices, test_accs, marker='o', color='darkcyan', label='test')
    plt.xlabel(label)
    plt.ylabel('average accuracy')
    plt.legend(loc='best')

    try:
    	plt.show()
    except:
    	return