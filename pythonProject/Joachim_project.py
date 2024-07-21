
# def forward_epoch(model, dl, loss_function, optimizer, total_loss=0,
#                   to_train=False, desc=None, device=torch.device('cpu'), weighting=False, bce2=False):
#
#     # total loss is over the entire epoch
#     # y_trues is by patient for the entire epoch; can get last batch with [-batch_size]
#     # y_preds is by patient for the entire epoch
#     #
#     with tqdm(total=len(dl), desc=desc, ncols=100) as pbar:
#         model = model.double().to(device)  # solving runtime memory issue
#
#         y_trues = torch.empty(0).type(torch.int).to(device)
#         y_preds = torch.empty(0).type(torch.int).to(device)
#         for i_batch, (X, y) in enumerate(dl):
#             # print('sickyall',X.dtype)
#             X = X.to(device)
#             X = X.type(torch.double)
#             # print('wackyall',X.dtype)
#             y = y.to(device)
#
#             # Forward:
#             y_pred = model(X)
#
#             # Loss:
#             y_true = y.type(torch.double)
#             if weighting: #loss_function.reduction == 'none':
#                 loss = bce2(y_pred, y_true)  # straight mean
#                 lossbatch = loss_function(y_pred, y_true)  # loss of one batch per batch element
#                 y_predn = y_pred.cpu().detach().numpy()
#                 y_truen = y_true.cpu().detach().numpy()
#                 lossn = lossbatch.cpu().detach().numpy()
#                 print()
#                 print(y_predn)
#                 print(y_truen) # TODO get weighting
#                 mask0 = ((y_predn != y_truen) + (y_truen == 0)) == 2
#                 mask1 = ((y_predn != y_truen) + (y_truen == 1)) == 2
#                 lossn[mask0] = weighting[0]*lossn[mask0]
#                 lossn[mask1] = weighting[1] * lossn[mask0]
#                 # multiply batch elements by weights
#                 loss[0] = lossn.mean()
#             else:
#                 loss = loss_function(y_pred, y_true)  # loss of one batch
#
#             total_loss += loss.item()  # added sum because reduction is zero and needs to be one ele to add item
#
#             y_trues = torch.cat((y_trues, y_true))
#             y_preds = torch.cat((y_preds, y_pred))
#             if to_train:
#                 # Backward:
#                 optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
#                 # optimizer.step()  # use gradients
#                 # for i in range(5):
#                     # optimizer.zero_grad()  # zero the gradients to not accumulate their changes.
#                     # loss[i].backward(retain_graph=True)  # get gradients
#                     # optimizer.step()  # use gradients
#                 loss.backward()  # get gradients
#
#                 # Optimization step:
#                 optimizer.step()  # use gradients
#
#             # Progress bar:
#             pbar.update(1)
#
#     return total_loss, y_trues, y_preds

#
# train_loss_vec = []
# test_loss_vec = []
# val_loss_vec = []
# train_acc_vec = []
# test_acc_vec = []
# val_acc_vec = []
# for i_epoch in range(num_epochs):
#     train_loss = 0
#     test_loss = 0
#     val_loss = 0
#
#     print(f'Epoch: {i_epoch + 1}/{num_epochs}')
#     # Train set
#     train_loss, y_true_train, y_pred_train = forward_epoch(model, train_loader, loss_function, optimizer,
#                                                            train_loss,
#                                                            to_train=True, desc='Train', device=device)
#     # Test set
#     test_loss, y_true_test, y_pred_test = forward_epoch(model, test_loader, loss_function, optimizer, test_loss,
#                                                         to_train=False, desc='Test', device=device)
#     # # Validation set
#     # val_loss, y_true_val, y_pred_val = forward_epoch(sex_net, dl_val, loss_function, optimizer, val_loss,
#     #                                                  to_train=False, desc='Validation', device=gpu_0, label=label)
#
#     # Metrics:
#     train_loss = train_loss / train_ds_size  # we want to get the mean over batches.
#     test_loss = test_loss / test_ds_size
#     # val_loss = val_loss / len(dl_val)
#     train_loss_vec.append(train_loss)
#     test_loss_vec.append(test_loss)
#     # val_loss_vec.append(val_loss)
#
#     # scikit-learn computations are numpy based; thus should run on CPU and without grads.
#     train_accuracy = accuracy_score(y_true_train.cpu(),
#                                     (y_pred_train.cpu().detach() > 0.5) * 1)
#     test_accuracy = accuracy_score(y_true_test.cpu(),
#                                    (y_pred_test.cpu().detach() > 0.5) * 1)
#     # val_accuracy = accuracy_score(y_true_val.cpu(),
#     #                               (y_pred_val.cpu().detach() > 0.5) * 1)
#     train_acc_vec.append(train_accuracy)
#     test_acc_vec.append(test_accuracy)
#     # val_acc_vec.append(val_accuracy)
#
#     print(f'train_loss={round(train_loss, 3)}; train_accuracy={round(train_accuracy, 3)} \
#           test_loss={round(test_loss, 3)}; test_accuracy={round(test_accuracy, 3)}')
# try:
#     if fn != 'None':
#         if fn[-7:] != ".pickle":
#             fn = fn + ".pickle"
#         torch.save(model.state_dict(), fn)
#         torch.save(model.state_dict(), fn[:-7]+'_opt'+fn[-7:])
#         #torch.save(sex_net, f=fn)
#         print('saved model')
# except:
#     print("didn't save")
#     pass
