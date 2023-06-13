import torch
from collections import defaultdict


class DequeDefaultDict(defaultdict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        defaultdict.__setitem__(self, key, value)
        if self.maxlen > 0:
            if len(self) > self.maxlen:
                self.pop(next(iter(self)))


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')



def attention_mask_full_mapet_1(permutation: torch.Tensor, num_heads: int, least_number: int, upper_limit: int = None):
    '''
    Function to extract query and content masks from data based on the permutation

    :param permutation:torch.Tensor permutation of the patches of an image. size: image_patches+1. Accounts also for classification_token
    :param num_heads: number of heads of the transformer
    :param least_number: least number of element to be considered in the query stream
    :return: single_mask_query (num_heads, sequence_lenght - least_number, sequence_lenght), single_mask_content (num_heads, sequence_lenght, sequence_lenght), index_vector (index of rows wich least_number)
    '''
    # N is the lenght of the sequence
    N = permutation.shape[-1]
    # permutation goes from 1 (classification token) to N+1
    # index_range is a column vector containing all the elements of the sequence from 0 to N
    index_range = torch.arange(0, N + 1, step=1).expand(1, -1).transpose(-2, -1)
    # result contains 2D indexes of where to attend (the colum [0] index is related to the row of the mask, column [1]
    #   index is related to the permutation value to look for
    # (permutation == index_range) create a tensor [N+1,N] which has True in corrispondance of where the element of
    #   index_range is equal with respect to the permutation. Each row has 1 True in corrispondance of the column_number
    #   of the element position in the permutation
    # (permutation == index_range).nonzero()[:, 1] allows to calculate the position in the permutation of each ordered
    #   element of index_range
    # (index_range < (permutation == index_range).nonzero()[:, 1]) column wise we have for each element of the
    #   index_range [0 to N-1] how many element before comes before itself on the permutation (marking them with True)
    #   escluding itself.
    #   For example if permutation is [1,5,8], column 4 (since it starts from 0) will have 1 true and all false and column
    #   column 8 will have 2 True and all false
    # torch.where(index_range < (permutation == index_range).nonzero()[:, 1], -1, 0) substitute true with -1
    # torch.where(index_range < (permutation == index_range).nonzero()[:, 1], -1, 0).transpose(-1, -2).nonzero()
    #   Transpose to get the view from rows and then
    result = torch.where(index_range < (permutation == index_range).nonzero()[:, 1], -1, 0).transpose(-1, -2).nonzero()

    # i is the vector of indexes that must be computed in order to calculate sparse tensor. -1 correct the fact that permutation starts from 1, but masks indexes start from 0
    i = torch.cat([torch.unsqueeze(result[:, 0], 0), torch.unsqueeze(permutation[result[:, 1]] - 1, 0)], 0)
    # sparse tensor takes three arguments: indexes, values and dimension
    # index are a 2D structure with first row the row of the values and second row the column of the values
    # since it's a mask values are just 1
    # dimension is related to the length of permutation
    s = torch.sparse_coo_tensor(i, torch.ones(result.shape[0]), (N, N))
    s = s.to_dense()
    # select_values indicate in wich row we attend more than least_number element as Boolean vector. It will be returned as index vector
    # select_values = torch.where(s.sum(dim=1) >= least_number, True, False)
    # least_values contains element that we don't want to predict --> ordered numerically
    least_values = s.sum(dim=1) < least_number
    upper_values = ~ (s.sum(dim=1) > upper_limit)
    # predict_values contains values that we want to predict ordered numerically
    predict_values = ~least_values
    # print(least_values)
    # print(predict_values)
    # print()
    # all the elements that we don't want to predict are evaluated
    s[:, least_values] = 1
    # s = torch.logical_and(non_functional_tokens, s)
    single_mask_content = s + torch.diag(torch.ones(N)).type(torch.DoubleTensor)

    # The mask of position is obtained with the negative column of the prediction indexes
    positional_mask = ~(s[:, predict_values]).type(torch.bool)
    s = torch.cat([s, positional_mask], dim=1)
    # The query stream only have the rows of the element to be predicted
    s = s[predict_values, :]
    # positiona mask evaluated on the content stream as the negative column of the prediction indexes
    positional_mask = ~(single_mask_content[:, predict_values]).type(torch.bool)
    single_mask_content = torch.cat([single_mask_content, positional_mask], dim=1)

    # attach a number of rows that is equal to the number of elements (Masked + position)
    num_predict_elements = torch.sum(predict_values)
    masked_tokens = torch.ones((num_predict_elements, s.shape[1]))
    masked_tokens[:, torch.cat((predict_values, torch.zeros(num_predict_elements)), dim=0).type(torch.bool)] = 0
    single_mask_content = torch.cat([single_mask_content, masked_tokens], dim=0)
    # single_mask_content = torch.logical_and(non_functional_tokens, single_mask_content)
    # single_mask_content[0][0] = 1
    # print("content",single_mask_content)
    # print("query",s)
    s = s.type(torch.DoubleTensor)
    single_mask_content = single_mask_content.type(torch.DoubleTensor)
    # print(s)
    # mask_query = torch.zeros((N,N)).type(torch.DoubleTensor)
    single_mask_query = torch.where(s == 1, 0.0, -1e30)

    single_mask_content = torch.where(single_mask_content == 0, -1e30, 0.0)
    # This was done to cut the query mask
    # single_mask_query = single_mask_query[least_values, :]
    single_mask_query = single_mask_query.expand(num_heads, -1, -1)
    single_mask_content = single_mask_content.expand(num_heads, -1, -1)

    return single_mask_query, single_mask_content, predict_values, upper_values



def main():
    N = 5
    rand = torch.rand(4, 5)
    batch_rand_perm = rand.argsort(dim=1)
    batch_rand_perm = batch_rand_perm + 1
    # print(batch_rand_perm)
    # permutation = torch.tensor([[3,2,5,4,1], [3,1,5,4,2]])
    permutation = torch.tensor([1, 3, 5, 4, 6, 2])
    # m_query, m_content = get_attention_mask(permutation, 1)

    n_head = 2
    least_number = 3
    m_query, m_content, vector_selected, upper_vector = attention_mask_upperlimit_fmapet(permutation, n_head, least_number, upper_limit=4)
    print(m_query)
    print(m_content)
    print(vector_selected)


if __name__ == "__main__":
    main()
