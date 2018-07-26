from torch.autograd import Variable

class validate(object):

    def __init__():
        self.name = 'validate'

    def __call__(devloader, model, crit, score_func, logpath):

        loss = 0.0
        score = 0.0

        for i, batch in enumerate(devloader):
            inputs, targets = batch

            outputs = model(inputs)

            loss += crit(outputs, targets)

            score += score_func(outputs.max(dim=1)[1].data.cpu().numpy()
                        ,targets.data.cpu().numpy())

        loss = loss/(i+1)
        score = score/(i+1)

        valid_log = '''
        [test] score: %.3f, loss: %.3f
        '''%(valid_score, valid_loss)

        os.system('echo %s >> %s'%(valid_log, logpath))
        print(valid_log)
        
        return loss/(i+1), score/(i+1)

class test(validate):
    def __init__():
        self.name = 'test'


def train(trainloader, valid_func, model, crit, score_func, optim, 
        lr, ephs, logpath):

    best_valid_score = 0.0
    best_model = model

    for epoch in range(ephs):

        for i, batch in enumerate(trainloader):

            inputs, targets = batch

            optim.zero_grad()

            model.train() # autograd on

            train_loss = crit(model(inputs), targets)

            train_log = '''

            [train] %5dth epoch, %5dth batch. loss: %.3f
            '''%(epoch, i, train_loss.data[0])
            
            os.system('echo %s >> %s'%(train_log, logpath))
            print(train_log)

            train_loss.backward()

            optim.step()

            model.eval() # autograd off

            valid_loss, valid_score = valid_func(model)


            if valid_score > best_valid_score:

                best_valid_uar = valid_uar
                state = {
                        'net_state_dict':model.state_dict(),
                        'train_loss':train_loss,
                        'valid_score':valid_score,
                        'optim_state_dict':optim.state_dict()}

                valid_log = '''
                [valid] bestscore: %.3f, loss: %.3f
                '''%(valid_score, valid_loss)
                os.system('echo %s >> %s'%(valid_log, logpath))

                best_model = model

            print('Finished Training')

            return best_model
