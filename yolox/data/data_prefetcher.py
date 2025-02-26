#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_input1,self.next_input2,self.next_tinput,self.next_tinput1,self.next_tinput2,self.next_target,_, self.id, self.id1 = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.id = None
            self.id1 = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        input1 = self.next_input1
        input2 = self.next_input2
        tinput = self.next_tinput
        tinput1 = self.next_tinput1
        tinput2 = self.next_tinput2
        target = self.next_target
        id = self.id[1]
        id1 = self.id1[1]
        if input is not None:
            self.record_stream(input)
            self.record_stream(input1)
            self.record_stream(input2)
            self.record_stream(tinput)
            self.record_stream(tinput1)
            self.record_stream(tinput2)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input,input1,input2,tinput,tinput1,tinput2, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)
        self.next_input1 = self.next_input1.cuda(non_blocking=True)
        self.next_input2 = self.next_input2.cuda(non_blocking=True)
        self.next_tinput = self.next_tinput.cuda(non_blocking=True)
        self.next_tinput1 = self.next_tinput1.cuda(non_blocking=True)
        self.next_tinput2 = self.next_tinput2.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

    @staticmethod
    def _record_stream_for_image(input1):
        input1.record_stream(torch.cuda.current_stream())

    @staticmethod
    def _record_stream_for_image(tinput):
        tinput.record_stream(torch.cuda.current_stream())

    @staticmethod
    def _record_stream_for_image(tinput1):
        tinput1.record_stream(torch.cuda.current_stream())