import pandas as pd


def texify(data):
    for i in range(len(data.axes[0])):
        date = data.axes[0][i]
        rowstr = '%02i/%02i/%i %02i:00' % \
            (date.month, date.day, date.year, date.hour)
        for j in range(len(data.axes[1])):
            rowstr += ' & %.2f' % data.ix[i, j]
        print rowstr + ' \\\\'

if __name__ == '__main__':
    asks = pd.read_csv('askdata.csv', parse_dates=True, index_col=0)
    bids = pd.read_csv('biddata.csv', parse_dates=True, index_col=0)

    print 'Asks:'
    texify(asks)
    print '\nBids'
    texify(bids)
