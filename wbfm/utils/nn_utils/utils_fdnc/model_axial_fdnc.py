# A version of the leifer utils_fdnc model using axial transformers instead of basic ones
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np
from wbfm.utils.external.transformers import RectAxialEncoder
from fDNC.src.model import PointTransFeat


class NITAxial(nn.Module):
    """ Simple Neuron Id transformer:
        - Transformer
    """

    def __init__(self, input_dim, n_hidden, n_layer=6,
                 length_to_pad=200,  # Pads all input to be length_to_pad x 3; should more more than the number of neurons
                 cuda=True,
                 p_rotate=False, feat_trans=False):
        """ Init Model

        """
        super(NITAxial, self).__init__()

        # self.p_rotate = p_rotate
        # self.feat_trans = feat_trans
        self.input_dim = input_dim  # Used?
        # self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.cuda = cuda
        self.length_to_pad = length_to_pad
        # self.fc_outlier = nn.Linear(n_hidden, 1)
        # NEW: no projection to hidden 1d layer
        # Linear Layer with bias, project 3d coordinate into hidden dimension.
        # self.point_f = PointTransFeat(rotate=self.p_rotate, feature_transform=feat_trans,
        #                               input_dim=input_dim, hidden_d=n_hidden)

        # NEW: axial encoder
        # self.enc_l = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=8)
        # self.model = nn.TransformerEncoder(self.enc_l, n_layer)
        self.model = RectAxialEncoder(in_channels=1, dim_h=length_to_pad, dim_w=3, blocks=n_hidden, heads=8)

        # TODO: pass device directly
        self.device = torch.device("cuda:0" if cuda else "cpu")

    def encode(self, pts_padded, pts_length):
        # pts_padded should be of (b, num_pts, 3)
        # pts_proj = self.h_projection(pts_padded)
        # pts_proj = self.point_f(pts_padded.transpose(2, 1))
        # pts_proj = pts_proj.transpose(2, 1)
        # mask = self.generate_sent_masks(pts_proj, pts_length)
        # add the src_key_mask need to test.
        # pts_encode = self.model(pts_proj.transpose(dim0=0, dim1=1), src_key_padding_mask=mask)
        pts_encode = self.model(pts_padded.transpose(dim0=0, dim1=1))

        return pts_encode.transpose(dim0=0, dim1=1)

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float).bool()
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = True
        return enc_masks.to(self.device)

    # def encode_pos(self, pts):
    #     """ Take a mini-batch of "source" and "target" points, compute the encoded hidden code for each
    #     neurons.
    #     @param pts1 (List[List[x, y, z]]): list of source points set
    #     @param pts2 (List[List[x, y, z]]): list of target points set
    #
    #     @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
    #                                 log-likelihood of generating the gold-standard target sentence for
    #                                 each example in the input batch. Here b = batch size.
    #     """
    #     # Compute sentence lengths
    #     # pts_lengths = [len(s) for s in pts]
    #     # pts_padded should be of dimension (
    #     pts_padded = self.to_input_tensor(pts)
    #
    #     pts_encode = self.encode(pts_padded, pts_length=None)
    #     return pts_encode

    def forward(self, pts, match_dict=None, ref_idx=0, mode='train'):
        """ Take a mini-batch of "source" and "target" points, compute the encoded hidden code for each
        neurons.
        @param pts1 (List[List[x, y, z]]): list of source points set
        @param pts2 (List[List[x, y, z]]): list of target points set

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        pts_lengths = [len(s) for s in pts]
        # pts_padded should be of dimension (
        # pts_padded = self.to_input_tensor(pts)
        pts_padded = self.pad_to_target_length(pts)

        pts_encode = self.encode(pts_padded, pts_lengths)

        ref_emb = pts_encode[ref_idx:ref_idx + 1, :pts_lengths[ref_idx], :]
        ref_emb = torch.repeat_interleave(ref_emb, repeats=pts_encode.size(0), dim=0)

        sim_m = torch.bmm(pts_encode, ref_emb.transpose(dim0=1, dim1=2))
        # the outlier of mov node
        mov_outlier = self.fc_outlier(pts_encode)
        sim_m = torch.cat((sim_m, mov_outlier), dim=2)

        p_m = F.log_softmax(sim_m, dim=2)
        p_m_exp = F.softmax(sim_m, dim=2)

        batch_sz = pts_encode.size(0)
        loss = 0
        num_pt = 0
        loss_entropy = 0
        num_unlabel = 0
        output_pairs = dict()

        if (mode == 'train') or (mode == 'all'):
            for i_w in range(batch_sz):
                # loss for labelled neurons.
                match = match_dict[i_w]
                if len(match) > 0:
                    match_mov = match[:, 0]
                    match_ref = match[:, 1]
                    log_p = p_m[i_w, match_mov, match_ref]
                    loss -= log_p.sum()
                    num_pt += len(match_mov)
                # loss for outliers.
                outlier_list = match_dict['outlier_{}'.format(i_w)]
                if len(outlier_list) > 0:
                    log_p_outlier = p_m[i_w, outlier_list, -1]
                    loss -= log_p_outlier.sum()
                    num_pt += len(outlier_list)

                # Entropy loss for unlabelled neurons.
                unlabel_list = match_dict['unlabel_{}'.format(i_w)]
                if len(unlabel_list) > 0:
                    loss_entropy_cur = p_m[i_w, unlabel_list, :] * p_m_exp[i_w, unlabel_list, :]
                    loss_entropy -= loss_entropy_cur.sum()
                    num_unlabel += len(unlabel_list)

        elif (mode == 'eval') or (mode == 'all'):
            output_pairs['p_m'] = p_m
            # choose the matched pairs for worm1
            paired_idx = torch.argmax(p_m, dim=1)
            output_pairs['paired_idx'] = paired_idx
            # TODO:
            # pick the maxima value

        loss_dict = dict()
        loss_dict['loss'] = loss
        loss_dict['num'] = num_pt if num_pt else 1
        loss_dict['loss_entropy'] = loss_entropy
        loss_dict['num_unlabel'] = num_unlabel if num_unlabel else 1

        loss_dict['reg_stn'] = self.point_f.reg_stn
        loss_dict['reg_fstn'] = self.point_f.reg_fstn

        return loss_dict, output_pairs

    def pad_to_target_length(self, pts, pad_pt=None):
        if pad_pt is None:
            pad_pt = [0, 0, 0]
        padding_var = []
        for s in pts:
            length_difference = self.length_to_pad - len(s)
            if length_difference > 0:
                padded = [pad_pt] * length_difference
                s = torch.tensor(torch.cat([s, padded]))
            elif length_difference < 0:
                s = s[:length_difference]
            padding_var.append(s)
            # padded[:len(s)] = s
            # padding_var.append(padded)
        output_tensor = torch.tensor(padding_var, dtype=torch.float, device=self.device)
        return output_tensor  # torch.t(sents_var)

    def to_input_tensor(self, pts, pad_pt=[0, 0, 0]):
        sents_padded = []
        max_len = max(len(s) for s in pts)
        for s in pts:
            padded = [pad_pt] * max_len
            padded[:len(s)] = s
            sents_padded.append(padded)
        # sents_var = torch.tensor(sents_padded, dtype=torch.long, device=self.device)
        sents_var = torch.tensor(sents_padded, dtype=torch.float, device=self.device)
        return sents_var  # torch.t(sents_var)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NITAxial(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        params = {
            'args': dict(input_dim=self.input_dim,
                         n_hidden=self.n_hidden,
                         n_layer=self.n_layer,
                         cuda=self.cuda),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class NITAxialRegistration(NITAxial):
    """ Neuron Id transformer for registration between two datasets:
        - Transformer
    """

    def __init__(self, input_dim, n_hidden, n_layer=6, cuda=True, p_rotate=False, feat_trans=False):
        super(NITAxialRegistration, self).__init__(input_dim, n_hidden, n_layer, cuda, p_rotate, feat_trans)
        self.point_f = PointTransFeat(rotate=self.p_rotate, feature_transform=feat_trans,
                                      input_dim=input_dim, hidden_d=n_hidden)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.fc_outlier = nn.Linear(n_hidden, 1)

    def encode(self, pts_padded, pts_length, ref_idx):
        pts_proj = self.point_f(pts_padded.transpose(2, 1))
        pts_proj = pts_proj.transpose(2, 1)
        mask = self.generate_sent_masks(pts_proj, pts_length)
        # append the ref points to each batch
        ref_pts_proj = pts_proj[ref_idx:ref_idx + 1, :pts_length[ref_idx], :]
        mask_ref = torch.zeros((mask.size(0), pts_length[ref_idx]), dtype=torch.float, device=mask.device).bool()

        # mov_ind = torch.zeros((pts_proj.size(0), pts_proj.size(1), 1), device=pts_proj.device)
        # ref_ind = torch.ones((pts_proj.size(0), pts_length[ref_idx], 1), device=pts_proj.device)
        #
        # pts_proj = torch.cat((pts_proj, mov_ind), dim=2)
        # ref_pts_proj = torch.cat((torch.repeat_interleave(ref_pts_proj, repeats=pts_proj.size(0), dim=0), ref_ind), dim=2)

        # simply add 1 to the ref
        ref_pts_proj = torch.repeat_interleave(ref_pts_proj, repeats=pts_proj.size(0), dim=0) + 1

        pts_proj = torch.cat((ref_pts_proj, pts_proj), dim=1)
        mask = torch.cat((mask_ref, mask), dim=1)

        pts_encode = self.model(pts_proj.transpose(dim0=0, dim1=1), src_key_padding_mask=mask)

        return pts_encode.transpose(dim0=0, dim1=1)

    def forward(self, pts, match_dict=None, ref_idx=0, mode='train'):
        """ Take a mini-batch of "source" and "target" points, compute the encoded hidden code for each
        neurons.
        @param pts1 (List[List[x, y, z]]): list of source points set
        @param pts2 (List[List[x, y, z]]): list of target points set

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        pts_lengths = [len(s) for s in pts]
        # pts_padded should be of dimension (
        pts_padded = self.to_input_tensor(pts)
        pts_encode = self.encode(pts_padded, pts_lengths, ref_idx)

        ref_emb = pts_encode[:, :pts_lengths[ref_idx], :]
        mov_emb = pts_encode[:, pts_lengths[ref_idx]:, :]

        sim_m = torch.bmm(mov_emb, ref_emb.transpose(dim0=1, dim1=2))
        # the outlier of mov node
        mov_outlier = self.fc_outlier(mov_emb)
        sim_m = torch.cat((sim_m, mov_outlier), dim=2)

        p_m = F.log_softmax(sim_m, dim=2)
        p_m_exp = F.softmax(sim_m, dim=2)

        batch_sz = pts_encode.size(0)
        loss = 0
        num_pt = 0
        loss_entropy = 0
        num_unlabel = 0
        output_pairs = dict()

        if (mode == 'train') or (mode == 'all'):
            for i_w in range(batch_sz):
                # loss for labelled neurons.
                match = match_dict[i_w]
                if len(match) > 0:
                    match_mov = match[:, 0]
                    match_ref = match[:, 1]
                    log_p = p_m[i_w, match_mov, match_ref]
                    loss -= log_p.sum()
                    num_pt += len(match_mov)
                # loss for outliers.
                outlier_list = match_dict['outlier_{}'.format(i_w)]
                if len(outlier_list) > 0:
                    log_p_outlier = p_m[i_w, outlier_list, -1]
                    loss -= log_p_outlier.sum()
                    num_pt += len(outlier_list)

                # Entropy loss for unlabelled neurons.
                unlabel_list = match_dict['unlabel_{}'.format(i_w)]
                if len(unlabel_list) > 0:
                    loss_entropy_cur = p_m[i_w, unlabel_list, :] * p_m_exp[i_w, unlabel_list, :]
                    loss_entropy -= loss_entropy_cur.sum()
                    num_unlabel += len(unlabel_list)

        elif (mode == 'eval') or (mode == 'all'):
            output_pairs['p_m'] = p_m
            # choose the matched pairs for worm1
            paired_idx = torch.argmax(p_m, dim=1)
            output_pairs['paired_idx'] = paired_idx
            # TODO:
            # pick the maxima value

        loss_dict = dict()
        loss_dict['loss'] = loss
        loss_dict['num'] = num_pt if num_pt else 1
        loss_dict['loss_entropy'] = loss_entropy
        loss_dict['num_unlabel'] = num_unlabel if num_unlabel else 1

        loss_dict['reg_stn'] = self.point_f.reg_stn
        loss_dict['reg_fstn'] = self.point_f.reg_fstn

        return loss_dict, output_pairs
