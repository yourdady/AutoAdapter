''' 
@project AutoAdapter
@author Peng
@file test.py
@time 2018-08-16
'''
from load_data import load_mnist, load_usps, parse_data
from AutoAdapter import autoAdapter

def test():
    ux, uy = parse_data()
    usps_data = load_usps(ux, uy, validation_size=5000, test_size=0)
    mnist_data = load_mnist(one_hot=True, validation_size=5000)

    kernel_param = {"alpha": 1.0, "d": 2, "c": 0.0}
    sigma_list = [50]
    aa = autoAdapter(input_dim=28*28, new_dim=100, n_classes=10,
                     batch_size_src=128, batch_size_tar=128,
                     training_steps=1000, lamb=0.01, kernel_type='linear',
                     kernel_param=kernel_param, sigma_list=sigma_list)
    aa.fit(mnist_data, usps_data, onehot=True, plot=True)
    # aa.transform(usps_data.dataset.validation.images)

if __name__ == '__main__':
    test()