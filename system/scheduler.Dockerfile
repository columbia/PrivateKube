FROM golang as builder

ENV GO111MODULE=on
ENV DIRPATH /app/columbia.github.com/privatekube

WORKDIR $DIRPATH
RUN mkdir privacyresource
RUN mkdir scheduler
RUN mkdir privacycontrollers

COPY privacyresource privacyresource/
COPY scheduler scheduler/
COPY privacycontrollers privacycontrollers/

WORKDIR $DIRPATH/privacyresource
RUN go mod download
RUN go mod tidy

WORKDIR $DIRPATH/privacycontrollers
RUN go mod download
RUN go mod tidy

WORKDIR $DIRPATH/scheduler
RUN go mod download
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux go build  -o ../scheduler -mod=mod

#second stage
FROM alpine:latest
ENV DIRPATH /app/columbia.github.com/privatekube
WORKDIR $DIRPATH

RUN apk add --no-cache tzdata
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder $DIRPATH .

ENTRYPOINT ["/app/columbia.github.com/privatekube/scheduler"]