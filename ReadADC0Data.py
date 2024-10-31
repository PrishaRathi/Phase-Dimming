# ____________________________________________________________________________
# -- Authored By: Prisha Rathi
# ____________________________________________________________________________
import numpy as np
from scipy import signal, int32
import matplotlib as mpl
import matplotlib.pyplot as plt

# -- Add current directory to sys.path and load our modules
import sys
import os

sys.path.append( os.getcwd( ) )


# -- Read the ADC0 Data into a ndarray
# -- Each line of data looks like =>
# --     HOT_VOLTS__ACVoltage_buf[0]     signed int  -7437   XRAM:0x400
def ReadADC0Data( DataFile ) :
    table = np.loadtxt( DataFile, dtype = {
        'names'   : ('BufVar', 'datatype-part1', 'datatype-part2', 'Value', 'Address'),
        'formats' : ('S64', 'S16', 'S16', 'i4', 'S16')
    } )
    Values = np.ndarray( len( table ), dtype = np.int32 )
    Values [ : ] = table [ 'Value' ]
    return Values


# -- Normalize the data buffer
def Normalize( DataBuf ) :
    minval = np.amin( DataBuf )
    maxval = np.amax( DataBuf )
    offset = minval + (maxval - minval) / 2
    
    # -- return normalized data
    return DataBuf - offset


# -- Calculate the FFTMagnitudes & Freq using integer math
def CalcFFTInt( ADC0DataBuf ) :
    # -- Add offset back to the data
    N = len( ADC0DataBuf )
    
    # -- Get the Window
    Window = signal.hamming( N )
    WindowedADC0Data = Window * ADC0DataBuf
    
    # -- Calculate FFT
    FFT = np.fft.rfft( WindowedADC0Data )
    bins = (N / 2) + 1
    FFTMag = np.ndarray( int32( bins ), dtype = np.int32 )
    FFTMag [ : ] = np.absolute( FFT )
    
    # -- Energy spectrum for all positive frequencies
    EnergySpectrum = int32( FFTMag [ :-1 ] )
    
    # -- Return the WindowedData & Energy Spectrum
    return WindowedADC0Data, EnergySpectrum


# -- Plot the Data
def PlotData( ADC0Data, NormalizedADC0Data, DimmingCnt, DimmedADC0Data, WindowedDataCalc, EnergySpectrumCalc ) :
    fig, axes = plt.subplots( nrows = 2, ncols = 3, figsize = (18, 12), squeeze = False )
    
    # -- Set our x axis
    t = np.linspace( 1, 64, 64 )
    
    # -- Set ADC0Data Axis
    axes [ 0, 0 ].set_title( 'ADC0 Measurements' )
    axes [ 0, 0 ].set_xlabel( 'sample #' )
    axes [ 0, 0 ].set_ylabel( 'adc0 value' )
    axes [ 0, 0 ].xaxis.set_major_locator( mpl.ticker.MultipleLocator( 16 ) )
    axes [ 0, 0 ].xaxis.set_minor_locator( mpl.ticker.MultipleLocator( 4 ) )
    axes [ 0, 0 ].grid( color = 'grey', which = 'major', axis = 'x', linestyle = '-', linewidth = 0.5 )
    axes [ 0, 0 ].grid( color = 'grey', which = 'minor', axis = 'x', linestyle = '-', linewidth = 0.2 )
    axes [ 0, 0 ].grid( color = 'grey', which = 'major', axis = 'y', linestyle = '-', linewidth = 0.5 )
    axes [ 0, 0 ].plot( t, ADC0Data )
    # -- Set Normalized ADC0Data Axis
    axes [ 0, 1 ].set_title( 'Normalized ADC0 Measurements' )
    axes [ 0, 1 ].set_xlabel( 'sample #' )
    axes [ 0, 1 ].set_ylabel( 'normalized adc0 value' )
    axes [ 0, 1 ].xaxis.set_major_locator( mpl.ticker.MultipleLocator( 16 ) )
    axes [ 0, 1 ].xaxis.set_minor_locator( mpl.ticker.MultipleLocator( 4 ) )
    axes [ 0, 1 ].grid( color = 'grey', which = 'major', axis = 'x', linestyle = '-', linewidth = 0.5 )
    axes [ 0, 1 ].grid( color = 'grey', which = 'minor', axis = 'x', linestyle = '-', linewidth = 0.2 )
    axes [ 0, 1 ].grid( color = 'grey', which = 'major', axis = 'y', linestyle = '-', linewidth = 0.5 )
    axes [ 0, 1 ].plot( t, NormalizedADC0Data )
    # -- Windowed ADC0Data Axis (Micro)
    axes [ 0, 2 ].set_title( 'After dimming [' + str( DimmingCnt * 100 / len( ADC0Data ) ) + '%]' )
    axes [ 0, 2 ].set_xlabel( 'sample #' )
    axes [ 0, 2 ].set_ylabel( 'windowed adc0 value' )
    axes [ 0, 2 ].xaxis.set_major_locator( mpl.ticker.MultipleLocator( 16 ) )
    axes [ 0, 2 ].xaxis.set_minor_locator( mpl.ticker.MultipleLocator( 4 ) )
    axes [ 0, 2 ].grid( color = 'grey', which = 'major', axis = 'x', linestyle = '-', linewidth = 0.5 )
    axes [ 0, 2 ].grid( color = 'grey', which = 'minor', axis = 'x', linestyle = '-', linewidth = 0.2 )
    axes [ 0, 2 ].grid( color = 'grey', which = 'major', axis = 'y', linestyle = '-', linewidth = 0.5 )
    axes [ 0, 2 ].plot( t, DimmedADC0Data )
    
    # -- Windowed ADC0Data Axis (PC)
    axes [ 1, 0 ].set_title( 'After applying Hamming Window' )
    axes [ 1, 0 ].set_xlabel( 'sample #' )
    axes [ 1, 0 ].set_ylabel( 'windowed adc0 value' )
    axes [ 1, 0 ].xaxis.set_major_locator( mpl.ticker.MultipleLocator( 16 ) )
    axes [ 1, 0 ].xaxis.set_minor_locator( mpl.ticker.MultipleLocator( 4 ) )
    axes [ 1, 0 ].grid( color = 'grey', which = 'major', axis = 'x', linestyle = '-', linewidth = 0.5 )
    axes [ 1, 0 ].grid( color = 'grey', which = 'minor', axis = 'x', linestyle = '-', linewidth = 0.2 )
    axes [ 1, 0 ].grid( color = 'grey', which = 'major', axis = 'y', linestyle = '-', linewidth = 0.5 )
    axes [ 1, 0 ].plot( t, WindowedDataCalc )
    # -- Set FFTMag Axis
    axes [ 1, 1 ].set_title( 'FFT Freq Spectrum' )
    axes [ 1, 1 ].set_xlabel( 'freq (Multiples of 60 Hz)' )
    axes [ 1, 1 ].set_ylabel( 'energy' )
    axes [ 1, 1 ].xaxis.set_major_locator( mpl.ticker.MultipleLocator( 10 ) )
    axes [ 1, 1 ].xaxis.set_minor_locator( mpl.ticker.MultipleLocator( 1 ) )
    axes [ 1, 1 ].bar( range( len( EnergySpectrumCalc ) ), EnergySpectrumCalc, width = 1.0 )
    
    # -- return our figure and axes
    return fig, axes


# -- Apply the dimming function
def Dimming( cnt, ADC0Data ) :
    DimmedADC0Data = np.ndarray( len( ADC0Data ), dtype = np.int32 )
    DimmedADC0Data [ : ] = ADC0Data
    for i in range( len( ADC0Data ) ) :
        if i < cnt :
            DimmedADC0Data [ i ] = 0
    
    # -- Return dimmed data
    return DimmedADC0Data


# -- Dimming Effect
def DimmingEffect( dimming, ADC0Buf, NormalizedADC0Buf ) :
    # -- Apply Dimming
    DimmedADC0Data = Dimming( dimming, NormalizedADC0Buf )
    
    # -- Now calculate FFT
    WCalc, ESCalc = CalcFFTInt( DimmedADC0Data )
    f, a = PlotData( ADC0Buf, NormalizedADC0Buf, dimming, DimmedADC0Data, WCalc, ESCalc )
    f.savefig( 'TestData/ADC0Data-' + str( dimming ) + '.png', dpi = 300 )


# -- Calculate the FFT
def main( ) :
    try :
        # -- Load the ADC0Data
        ADC0DataFile = 'TestData/ADC0Data.txt'
        
        # -- Read data from the files
        ADC0Buf = ReadADC0Data( ADC0DataFile )
        # -- Normalize the measurements
        NormalizedADC0Buf = Normalize( ADC0Buf )
        
        # -- Apply Dimming in steps of 4
        for i in range( len( ADC0Buf ) ) :
            if i % 4 == 0 :
                DimmingEffect( i, ADC0Buf, NormalizedADC0Buf )
    
    except Exception as e :
        print( e )


if __name__ == '__main__' :
    main( )
    
    # -- Display the Plots
    # -- plt.ion( )
    # -- plt.show( block = True )
