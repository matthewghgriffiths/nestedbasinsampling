

module nestedsamplingutils

implicit none

contains

function logaddexp(x1, x2) result(res)

implicit none

double precision x1, x2
double precision x_max, res

x_max = max(x1, x2)
res = log(exp(x1 - x_max) + exp(x2 - x_max)) + x_max

end

subroutine combineruns(emaxs, nlives, nruns, ndead, &
 &                     Estart, Efinish, emaxnew, nlivenew, ndeadnew, info)
implicit none

integer, intent(in) :: nruns, ndead
integer, intent(in) :: nlives(nruns, 0:ndead-1)
double precision, intent(in) :: emaxs(nruns, 0:ndead-1), Estart, Efinish
double precision, intent(out) :: emaxnew(0:nruns*ndead-1)
integer, intent(out) :: nlivenew(0:nruns*ndead-1), ndeadnew, info

double precision Ecurr, Emaxcurr(nruns)
integer currind(nruns), nactive(nruns)
integer i, j, k, nlive

Ecurr = Estart

do i=1, nruns
    do j=0,ndead-1
        if (emaxs(i,j).lt.Ecurr) exit
    enddo
    currind(i) = j
    nactive(i) = nlives(i,j)
    Emaxcurr(i) = emaxs(i,j)
enddo

nlive = sum(nactive)
i = maxloc(Emaxcurr, 1)
Ecurr = Emaxcurr(i)

ndeadnew = 0
do while(Ecurr.ge.Efinish)

    emaxnew(ndeadnew) = Ecurr
    nlivenew(ndeadnew) = nlive
    ndeadnew = ndeadnew + 1

    currind(i) = currind(i) + 1
    nlive = nlive - nactive(i)

    if ((currind(i).eq.ndead).or.(nlives(i,currind(i)).eq.0)) then
        Emaxcurr(i) = - huge(1.d0)
        nactive(i) = 0
        i = maxloc(Emaxcurr, 1)
        do while((Emaxcurr(i).eq.Ecurr).and.(currind(i).eq.0))
            currind(i) = 1
            Emaxcurr(i) = Emaxs(i,1)
            nactive(i) = nlives(i,1)
            nlive = nlive + nactive(i)
            i = maxloc(Emaxcurr, 1)
        end do
    else
        Emaxcurr(i) = Emaxs(i,currind(i))
        nactive(i) = nlives(i,currind(i))
        nlive = nlive + nactive(i)
    end if

    if (all(nactive.eq.0)) exit

    i = maxloc(Emaxcurr, 1)
    do while(currind(i).eq.0)
        currind(i) = 1
        Emaxcurr(i) = Emaxs(i,1)
        nactive(i) = nlives(i,1)
        nlive = nlive + nactive(i)
        i = maxloc(Emaxcurr, 1)
    enddo

    i = maxloc(Emaxcurr, 1)
    Ecurr = Emaxcurr(i)

end do

end subroutine combineruns

end module nestedsamplingutils
