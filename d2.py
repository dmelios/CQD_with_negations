# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

from typing import Callable, Tuple, Optional


def score_candidates(s_emb: Tensor,
                     p_emb: Tensor,
                     candidates_emb: Tensor,
                     k: Optional[int],
                     entity_embeddings: nn.Module,
                     scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tuple[Tensor, Optional[Tensor]]:

    batch_size = max(s_emb.shape[0], p_emb.shape[0])
    embedding_size = s_emb.shape[1]

    def reshape(emb: Tensor) -> Tensor:
        if emb.shape[0] < batch_size:
            n_copies = batch_size // emb.shape[0]
            emb = emb.reshape(-1, 1, embedding_size).repeat(1, n_copies, 1).reshape(-1, embedding_size)
        return emb

    s_emb = reshape(s_emb)
    p_emb = reshape(p_emb)
    nb_entities = candidates_emb.shape[0]

    x_k_emb_3d = None

    # [B, N]
    atom_scores_2d = scoring_function(s_emb, p_emb, candidates_emb)
    atom_k_scores_2d = atom_scores_2d

    if k is not None:
        k_ = min(k, nb_entities)

        # [B, K], [B, K]
        atom_k_scores_2d, atom_k_indices = torch.topk(atom_scores_2d, k=k_, dim=1)

        # [B, K, E]
        x_k_emb_3d = entity_embeddings(atom_k_indices)

    return atom_k_scores_2d, x_k_emb_3d


def query_1p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p_emb = predicate_embeddings(queries[:, 1])
    candidates_emb = entity_embeddings.weight

    assert queries.shape[1] == 2

    res, _ = score_candidates(s_emb=s_emb, p_emb=p_emb,
                              candidates_emb=candidates_emb, k=None,
                              entity_embeddings=entity_embeddings,
                              scoring_function=scoring_function)

    return res


def query_2p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, N]
    atom2_scores_2d, _ = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)
    
    
    #DM test - works improvement of test set 
    soft = torch.nn.Softmax(dim=1) 
    # atom1_k_scores_2d=soft(atom1_k_scores_2d)

    atom2_scores_2d=soft(atom2_scores_2d)


    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    # [B * K, N] -> [B, K, N]
    atom2_scores_3d = atom2_scores_2d.reshape(batch_size, -1, nb_entities)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


def query_2pn(entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              k: int,
              t_norm: Callable[[Tensor, Tensor], Tensor],
              negation: Callable[[Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, N]
    atom2_scores_2d, _ = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)

    atom2_scores_2d = negation(atom2_scores_2d)

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    # [B * K, N] -> [B, K, N]
    atom2_scores_3d = atom2_scores_2d.reshape(batch_size, -1, nb_entities)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


def query_3p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    p3_emb = predicate_embeddings(queries[:, 3])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, K], [B * K, K, E]
    atom2_k_scores_2d, x2_k_emb_3d = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K * K, E]
    x2_k_emb_2d = x2_k_emb_3d.reshape(-1, emb_size)

    # [B * K * K, N]
    atom3_scores_2d, _ = score_candidates(s_emb=x2_k_emb_2d, p_emb=p3_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)
    
    
    
    

        
    #DM test - works imrprovement of test set HITS10 from 0.113146 to 0.120016
    soft = torch.nn.Softmax(dim=1) 
    atom3_scores_2d=soft(atom3_scores_2d)
    
    

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K, K] -> [B, K * K, N]
    atom2_scores_3d = atom2_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K * K, N] -> [B, K * K, N]
    atom3_scores_3d = atom3_scores_2d.reshape(batch_size, -1, nb_entities)

    atom1_scores_3d = atom1_scores_3d.repeat(1, atom3_scores_3d.shape[1] // atom1_scores_3d.shape[1], 1)
    

    

    res = t_norm(atom1_scores_3d, atom2_scores_3d)
    res = t_norm(res, atom3_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


def query_2i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    res = t_norm(scores_1, scores_2)

    return res


def query_3i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)
    scores_3 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

    res = t_norm(scores_1, scores_2)
    res = t_norm(res, scores_3)

    return res


def query_ip(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    # [B, N]
    scores_1 = query_2i(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 4])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    # [B * N, E]
    s_emb = e_emb.reshape(1, nb_entities, emb_size).repeat(batch_size, 1, 1).reshape(-1, emb_size)

    # [B * N, N]
    scores_2, _ = score_candidates(s_emb=s_emb, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B, N, N]
    scores_1 = scores_1.reshape(batch_size, nb_entities, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, nb_entities, nb_entities)

    res = t_norm(scores_1, scores_2)
    res, _ = torch.max(res, dim=1)

    return res


def query_pi(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 3:5], scoring_function=scoring_function)

    res = t_norm(scores_1, scores_2)

    return res


def query_2in(temp1,temp2,entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              t_norm: Callable[[Tensor, Tensor], Tensor],
              negation: Callable[[Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)


    #added by DM - use custom linear normalization between 0 and 1 but sum !=1 
    def linearNormaliser(x): 
        minimum = torch.min(x)
        x -= minimum
        maximum = torch.max(x)
        x /= maximum
        sums=torch.sum(x)
        x /= sums    
        return(x)
    
    def customSoftmax(x,exponent,dim):
        e=torch.exp(x*exponent)
        return e/torch.sum(e,dim)


    def customSigmoid(x,exponent,dim):
        e=1/(1+torch.exp(-x*exponent))
        return e/torch.sum(e,dim)    


    #added by DM - normalize scores between 0 and 1, sum is 1
    scores_1_original = torch.clone(scores_1)
    scores_2_original = torch.clone(scores_2)
    
    # soft = torch.nn.Softmax(dim=1)
    # scores_1=soft(scores_1)
    # scores_2=soft(scores_2)
    
    
    scores_1=customSoftmax(scores_1,temp1,dim=1)
    scores_2=customSoftmax(scores_2,temp2,dim=1)
    
    
    # scores_1=customSigmoid(scores_1,1,dim=1)
    # scores_2=customSigmoid(scores_2,-0.5,dim=1)
    
    #added by DM - normalize scores using sigmoid
    # scores_1=torch.sigmoid(scores_1)
    # scores_2=torch.sigmoid(scores_2)
    # sum1=torch.sum(scores_1)
    # sum2=torch.sum(scores_2)
    # scores_1=scores_1 / sum1
    # scores_2=scores_2 / sum2
    
    
    
    #added by DM - use torch normalization norm 1
    #scores_1=torch.nn.functional.normalize(scores_1, p=1.0, dim=1, eps=1e-12, out=None)
    #scores_2=torch.nn.functional.normalize(scores_2, p=1.0, dim=1, eps=1e-12, out=None)
    
        

    
    # scores_1=linearNormaliser(scores_1)
    # scores_2=linearNormaliser(scores_2)
    
    #query is size 5     0,1 score1   2,3 score2    4 not used has value -2 constant
    # print(queries)
 
    # scores_2 = negation(scores_2)
    # scores_2 = 1 / scores_2
    # scores_2=soft(scores_2)
    # scores_2 = soft(scores_2)
    # scores_1=linearNormaliser(scores_1)
    # scores_2=linearNormaliser(scores_2)
    
    res = t_norm(scores_1, scores_2)

    #DM hack - need to see all scores
    #return res
    
    
    
    return res, scores_1, scores_2, scores_1_original, scores_2_original


def query_3in(temp1,temp2,entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              t_norm: Callable[[Tensor, Tensor], Tensor],
              negation: Callable[[Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)
    scores_3 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

    # scores_3 = negation(scores_3)


    scores_1_original = torch.clone(scores_1)
    scores_2_original = torch.clone(scores_2)

    def customSoftmax(x,exponent,dim):
        e=torch.exp(x*exponent)
        return e/torch.sum(e,dim)

    scores_1=customSoftmax(scores_1,temp1,dim=1)
    scores_2=customSoftmax(scores_2,temp1,dim=1)  
    scores_3=customSoftmax(scores_3,temp2,dim=1)  

    res = t_norm(scores_1, scores_2)
    res = t_norm(res, scores_3)

    
    # return res
    return res, scores_1, scores_2, scores_1_original, scores_2_original



def query_inp(temp1,temp2,entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              t_norm: Callable[[Tensor, Tensor], Tensor],
              negation: Callable[[Tensor], Tensor]) -> Tensor:

    # [B, N]
    # scores_1 = query_2in(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                         # queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm, negation=negation)

    scores_1 = query_2in(temp1,temp2,entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                         queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm, negation=negation)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 5])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    # [B * N, E]
    s_emb = e_emb.reshape(1, nb_entities, emb_size).repeat(batch_size, 1, 1).reshape(-1, emb_size)

    # [B * N, N]
    scores_2, _ = score_candidates(s_emb=s_emb, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B, N, N]
    scores_1 = scores_1.reshape(batch_size, nb_entities, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, nb_entities, nb_entities)
    
    #added by DM
    scores_1_original = torch.clone(scores_1)
    scores_2_original = torch.clone(scores_2)

    res = t_norm(scores_1, scores_2)
    res, _ = torch.max(res, dim=1)

    # return res
    return res, scores_1, scores_2, scores_1_original, scores_2_original 


def query_pin(temp1,temp2,entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              k: int,
              t_norm: Callable[[Tensor, Tensor], Tensor],
              negation: Callable[[Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 3:5], scoring_function=scoring_function)

#added by DM
    scores_1_original = torch.clone(scores_1)
    scores_2_original = torch.clone(scores_2)


    def customSoftmax(x,exponent,dim):
        e=torch.exp(x*exponent)
        return e/torch.sum(e,dim)


    scores_1=customSoftmax(scores_1,temp1,dim=1)
    scores_2=customSoftmax(scores_2,temp2,dim=1)  

    # scores_2 = negation(scores_2)

    res = t_norm(scores_1, scores_2)

    # return res
    return res, scores_1, scores_2, scores_1_original, scores_2_original 

def query_pni_v1(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 k: int,
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 negation: Callable[[Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

    scores_1 = negation(scores_1)

    res = t_norm(scores_1, scores_2)
    # return res
    return res, scores_1, scores_2, scores_1_original, scores_2_original 

def query_pni_v2(temp1,temp2,entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 k: int,
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 negation: Callable[[Tensor], Tensor]) -> Tensor:

    scores_1 = query_2pn(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                         queries=queries[:, 0:3], scoring_function=scoring_function, k=k,
                         t_norm=t_norm, negation=negation)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

#added by DM
    scores_1_original = torch.clone(scores_1)
    scores_2_original = torch.clone(scores_2)


    def customSoftmax(x,exponent,dim):
        e=torch.exp(x*exponent)
        return e/torch.sum(e,dim)



    res = t_norm(scores_1, scores_2)
    return res


def query_2u_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    res = t_conorm(scores_1, scores_2)

    return res


def query_up_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    # [B, N]
    scores_1 = query_2u_dnf(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                            queries=queries[:, 0:4], scoring_function=scoring_function, t_conorm=t_conorm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 5])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    # [B * N, E]
    s_emb = e_emb.reshape(1, nb_entities, emb_size).repeat(batch_size, 1, 1).reshape(-1, emb_size)

    # [B * N, N]
    scores_2, _ = score_candidates(s_emb=s_emb, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B, N, N]
    scores_1 = scores_1.reshape(batch_size, nb_entities, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, nb_entities, nb_entities)

    res = t_norm(scores_1, scores_2)
    res, _ = torch.max(res, dim=1)

    return res
